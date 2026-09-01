#!/usr/bin/env python3
"""High-level checkpoint mission controller for the OmniVLA edge node.

Waypoints are generated using A* path planning on the static costmap if available,
falling back to linear GPS interpolation if no costmap is present. This avoids
untraversable terrain (buildings, obstacles) when planning routes to checkpoints.
"""

import heapq
import json
import math
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import requests

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32

from erc_inference_msgs.action import StartMission
from erc_static_map.erc_static_map_node import utm_crs_for, project_point_to_utm


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: int
    sequence: int
    latitude: float
    longitude: float


class CheckpointControllerNode(Node):
    # 8-connected A* neighbor offsets (dcol, drow, step_cost)
    _SQRT2 = math.sqrt(2.0)
    _NEIGHBORS = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
    ]
    LETHAL_COST = 100  # Building/obstacle cost from OccupancyGrid

    def __init__(self):
        super().__init__('checkpoint_controller_node')

        self.declare_parameter('checkpoint_list_url', 'http://localhost:8000/checkpoints-list')
        self.declare_parameter('checkpoint_reached_url', 'http://localhost:8000/checkpoint-reached')
        self.declare_parameter('gps_topic', '/erc/gps')
        self.declare_parameter('model_cmd_vel_topic', '/omnivla/cmd_vel')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('goal_gps_topic', '/goal_gps')
        self.declare_parameter('goal_compass_topic', '/goal_compass')
        self.declare_parameter('use_pose_goal_topic', '/use_pose_goal')
        self.declare_parameter('use_satellite_topic', '/use_satellite')
        self.declare_parameter('use_image_goal_topic', '/use_image_goal')
        self.declare_parameter('use_lan_prompt_topic', '/use_lan_prompt')
        self.declare_parameter('enable_inference_topic', '/enable_inference')
        self.declare_parameter('proximity_threshold_m', 8.0)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('http_timeout_s', 15.0)
        self.declare_parameter('verification_retry_s', 1.0)

        self._lock = threading.RLock()
        self._gps_condition = threading.Condition(self._lock)
        self._current_lat: Optional[float] = None
        self._current_lon: Optional[float] = None
        # Latched from the first /erc/gps fix. This must be the SAME point
        # used as origin_lat/origin_lon when erc_static_map_node was launched
        # for this leg -- that call is what the published costmap's local-ENU
        # frame and UTM zone are anchored to, and the OccupancyGrid message
        # carries no CRS/absolute-position metadata to recover it from later.
        # Latching on first fix (rather than re-reading self._current_lat at
        # planning time) keeps it fixed for the whole leg even if the rover
        # has since moved, matching erc_static_map_node's fixed-A convention.
        self._map_origin: Optional[Tuple[float, float]] = None
        self._last_model_cmd = Twist()
        self._mission_active = False
        self._motion_allowed = False
        self._stop_requested = True
        self._current_sequence = 0
        self._current_distance_m = float('inf')
        self._costmap: Optional[OccupancyGrid] = None
        self._costmap_lock = threading.Lock()

        self._model_cmd_sub = self.create_subscription(
            Twist, self._param('model_cmd_vel_topic'), self._model_cmd_callback, 10
        )
        self._gps_sub = self.create_subscription(
            NavSatFix, self._param('gps_topic'), self._gps_callback, 10
        )
        self._costmap_sub = self.create_subscription(
            OccupancyGrid, 'erc_static_map/costmap', self._costmap_callback, 10
        )

        self._cmd_pub = self.create_publisher(Twist, self._param('cmd_vel_topic'), 10)
        self._goal_gps_pub = self.create_publisher(NavSatFix, self._param('goal_gps_topic'), 10)
        self._goal_compass_pub = self.create_publisher(Float32, self._param('goal_compass_topic'), 10)
        self._use_pose_pub = self.create_publisher(Bool, self._param('use_pose_goal_topic'), 10)
        self._use_satellite_pub = self.create_publisher(Bool, self._param('use_satellite_topic'), 10)
        self._use_image_pub = self.create_publisher(Bool, self._param('use_image_goal_topic'), 10)
        self._use_lan_pub = self.create_publisher(Bool, self._param('use_lan_prompt_topic'), 10)
        self._enable_pub = self.create_publisher(Bool, self._param('enable_inference_topic'), 10)

        self._action_server = ActionServer(
            self,
            StartMission,
            'start_mission',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )
        self._timer = self.create_timer(1.0 / self._param('control_rate_hz'), self._publish_output)
        self.get_logger().info('Checkpoint controller ready; waiting for start_mission action.')

    def _param(self, name: str):
        return self.get_parameter(name).value

    def _goal_callback(self, _goal_request):
        with self._lock:
            return GoalResponse.REJECT if self._mission_active else GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _model_cmd_callback(self, msg: Twist):
        with self._lock:
            self._last_model_cmd = msg

    def _gps_callback(self, msg: NavSatFix):
        with self._gps_condition:
            self._current_lat = msg.latitude
            self._current_lon = msg.longitude
            if self._map_origin is None:
                # Reject the SDK's "no lock yet" placeholder (0, 0) -- same
                # check used for the declination bootstrap fix in
                # erc_localization/launch/localization_global.launch.py.
                # Latching here means this node's map origin is whatever the
                # rover's actual first fix was; erc_static_map_node must be
                # launched with that SAME lat/lon as its origin_lat/origin_lon
                # for the costmap frame to line up.
                if abs(msg.latitude) > 1e-6 or abs(msg.longitude) > 1e-6:
                    self._map_origin = (msg.latitude, msg.longitude)
                    self.get_logger().info(
                        f'Latched map origin from first GPS fix: '
                        f'({msg.latitude:.8f}, {msg.longitude:.8f})')
            self._gps_condition.notify_all()

    def _costmap_callback(self, msg: OccupancyGrid):
        """Store the latest costmap for A* waypoint generation."""
        with self._costmap_lock:
            self._costmap = msg
            self.get_logger().debug(f'Updated static costmap: {msg.info.width}x{msg.info.height} @ {msg.info.resolution}m/cell')

    def _publish_output(self):
        with self._lock:
            output = self._last_model_cmd if (
                self._mission_active and self._motion_allowed and not self._stop_requested
            ) else Twist()
        self._cmd_pub.publish(output)

    def _publish_goal(self, checkpoint: Checkpoint):
        goal = NavSatFix()
        goal.latitude = checkpoint.latitude
        goal.longitude = checkpoint.longitude
        self._goal_gps_pub.publish(goal)

        self._goal_compass_pub.publish(Float32(data=0.0))
        self._use_pose_pub.publish(Bool(data=True))
        self._use_satellite_pub.publish(Bool(data=False))
        self._use_image_pub.publish(Bool(data=False))
        self._use_lan_pub.publish(Bool(data=False))
        self._enable_pub.publish(Bool(data=True))

        with self._lock:
            self._motion_allowed = True
            self._stop_requested = False
            self._current_sequence = checkpoint.sequence

    def _request_stop(self):
        with self._lock:
            self._motion_allowed = False
            self._stop_requested = True
        self._enable_pub.publish(Bool(data=False))
        self._publish_zero()

    def _resume_motion(self):
        with self._lock:
            self._motion_allowed = True
            self._stop_requested = False
        self._enable_pub.publish(Bool(data=True))

    def _publish_zero(self):
        self._cmd_pub.publish(Twist())

    @staticmethod
    def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        # Equirectangular approximation is accurate enough for a 5 m gate.
        earth_radius_m = 6378137.0
        lat0 = (lat1 + lat2) * 0.5
        d_lat = (lat2 - lat1) * 0.017453292519943295
        d_lon = (lon2 - lon1) * 0.017453292519943295
        north = d_lat * earth_radius_m
        east = d_lon * earth_radius_m * math.cos(lat0 * 0.017453292519943295)
        return (north * north + east * east) ** 0.5

    def _distance_to(self, checkpoint: Checkpoint) -> float:
        with self._lock:
            if self._current_lat is None or self._current_lon is None:
                return float('inf')
            return self._distance_m(
                self._current_lat,
                self._current_lon,
                checkpoint.latitude,
                checkpoint.longitude,
            )

    @staticmethod
    def _interpolate_gps(lat1: float, lon1: float, lat2: float, lon2: float, fraction: float):
        """Interpolate along the geodesic path from (lat1, lon1) to (lat2, lon2).
        fraction: 0.0 = start, 1.0 = end. Uses equirectangular approximation."""
        return (
            lat1 + (lat2 - lat1) * fraction,
            lon1 + (lon2 - lon1) * fraction,
        )

    @staticmethod
    def _octile_heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Octile distance heuristic for A* (allows 8-connected movement)."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (CheckpointControllerNode._SQRT2 - 2.0) * min(dx, dy)

    @staticmethod
    def _astar_grid(occupied: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """8-connected A* on a 2D grid. occupied: True = blocked, indexed [row, col].
        start/goal: (col, row). Returns list of (col, row) from start to goal, or None if no path."""
        height, width = occupied.shape

        def in_bounds(c: int, r: int) -> bool:
            return 0 <= c < width and 0 <= r < height

        if not in_bounds(*start) or not in_bounds(*goal):
            return None
        if occupied[start[1], start[0]] or occupied[goal[1], goal[0]]:
            return None

        open_heap = [(0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            closed.add(current)

            cc, cr = current
            for dc, dr, step_cost in CheckpointControllerNode._NEIGHBORS:
                nc, nr = cc + dc, cr + dr
                if not in_bounds(nc, nr) or occupied[nr, nc]:
                    continue
                # Prevent diagonal cuts through lethal corners
                if dc != 0 and dr != 0:
                    if occupied[cr, cc + dc] or occupied[cr + dr, cc]:
                        continue
                neighbor = (nc, nr)
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + CheckpointControllerNode._octile_heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))
        return None

    def _generate_waypoints_from_path(self, cell_path: List[Tuple[int, int]],
                                       utm_crs: str, origin_utm: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Sample 10 waypoints evenly along a cell path and convert to GPS.

        utm_crs / origin_utm are the SAME values used to build the grid this
        cell_path was planned over (see _generate_waypoints), so this is a
        straight inverse of erc_static_map_node's cell_to_world -> local-ENU
        -> UTM -> GPS chain rather than a guess.
        """
        import pyproj

        if not cell_path or len(cell_path) < 2:
            return []

        grid = self._costmap
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        resolution = grid.info.resolution
        origin_x, origin_y = origin_utm
        transformer = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

        def cell_to_world(cell: Tuple[int, int]) -> Tuple[float, float]:
            c, r = cell
            return (ox + (c + 0.5) * resolution, oy + (r + 0.5) * resolution)

        def world_to_gps(wx: float, wy: float) -> Tuple[float, float]:
            """Convert local ENU (m, relative to origin_utm) back to GPS.

            erc_static_map_node publishes grid.info.origin as
            (min_utm - origin_utm), i.e. the grid is already in a frame
            relative to origin_utm -- so recovering an absolute UTM position
            means ADDING origin_utm back, not treating (wx, wy) as if it were
            already absolute UTM.
            """
            utm_x = wx + origin_x
            utm_y = wy + origin_y
            lon, lat = transformer.transform(utm_x, utm_y)
            return lat, lon

        # Sample 10 waypoints evenly spaced along the path
        waypoints_gps = []
        for i in range(1, 11):
            idx = int(len(cell_path) * i / 10.0)
            idx = min(idx, len(cell_path) - 1)
            cell = cell_path[idx]
            wx, wy = cell_to_world(cell)
            waypoints_gps.append(world_to_gps(wx, wy))
        return waypoints_gps

    def _generate_waypoints(self, start_lat: float, start_lon: float, target_lat: float, target_lon: float) -> list:
        """Generate 10 GPS waypoints using A* if costmap available, else linear interpolation.
        Avoids untraversable terrain (buildings/obstacles) when costmap is present.

        The costmap's local-ENU frame and UTM zone are anchored to
        self._map_origin (latched from the first /erc/gps fix -- see
        _gps_callback). erc_static_map_node must have been launched with that
        SAME lat/lon as its origin_lat/origin_lon for this leg. All GPS<->grid
        conversions here go through that same UTM projection, mirroring
        erc_astar_planner_node, instead of assuming the rover is at the grid
        origin or using a flat degrees-to-meters scale that ignores the
        cos(latitude) correction on longitude.
        """
        with self._lock:
            map_origin = self._map_origin

        with self._costmap_lock:
            if self._costmap is not None and map_origin is not None:
                try:
                    map_origin_lat, map_origin_lon = map_origin
                    grid = self._costmap
                    ox = grid.info.origin.position.x
                    oy = grid.info.origin.position.y
                    resolution = grid.info.resolution
                    width = grid.info.width
                    height = grid.info.height

                    # Parse occupancy grid
                    occupied = (np.array(grid.data, dtype=np.int16).reshape((height, width)) >= self.LETHAL_COST)

                    def world_to_cell(x: float, y: float) -> Tuple[int, int]:
                        return (int(math.floor((x - ox) / resolution)),
                                int(math.floor((y - oy) / resolution)))

                    # Same UTM zone/projection erc_static_map_node used to build
                    # this grid, derived from the same latched map origin -- not
                    # a hardcoded zone and not the rover's current position.
                    utm_crs = utm_crs_for(map_origin_lat, map_origin_lon)
                    origin_utm = project_point_to_utm(map_origin_lat, map_origin_lon, utm_crs)
                    start_utm = project_point_to_utm(start_lat, start_lon, utm_crs)
                    goal_utm = project_point_to_utm(target_lat, target_lon, utm_crs)

                    # Local ENU relative to the map origin, matching
                    # erc_static_map_node's costmap_to_occupancy_grid convention.
                    start_local = (start_utm[0] - origin_utm[0], start_utm[1] - origin_utm[1])
                    goal_local = (goal_utm[0] - origin_utm[0], goal_utm[1] - origin_utm[1])

                    start_cell = world_to_cell(*start_local)
                    goal_cell = world_to_cell(*goal_local)

                    # Run A* on the grid
                    cell_path = self._astar_grid(occupied, start_cell, goal_cell)
                    if cell_path and len(cell_path) > 1:
                        waypoints = self._generate_waypoints_from_path(cell_path, utm_crs, origin_utm)
                        if waypoints:
                            self.get_logger().info(f'Generated {len(waypoints)} A* waypoints to avoid obstacles')
                            return waypoints
                except Exception as e:
                    self.get_logger().warn(f'A* waypoint generation failed ({e}); using linear interpolation')
            elif self._costmap is not None:
                self.get_logger().warn(
                    'Costmap available but no GPS fix has been latched as map origin yet; '
                    'skipping A* and using linear interpolation instead of guessing the UTM zone.')

        # Fallback: linear GPS interpolation at 10% intervals
        waypoints = []
        for i in range(1, 11):
            fraction = i / 10.0
            lat, lon = self._interpolate_gps(
                start_lat, start_lon, target_lat, target_lon, fraction
            )
            waypoints.append((lat, lon))
        return waypoints

    def _fetch_checkpoints(self):
        response = requests.post(
            self._param('checkpoint_list_url'), json={}, timeout=self._param('http_timeout_s')
        )
        response.raise_for_status()
        payload = response.json()
        checkpoints = [
            Checkpoint(
                checkpoint_id=int(item['id']),
                sequence=int(item['sequence']),
                latitude=float(item['latitude']),
                longitude=float(item['longitude']),
            )
            for item in payload['checkpoints_list']
        ]
        checkpoints.sort(key=lambda item: item.sequence)
        return checkpoints, int(payload.get('latest_scanned_checkpoint', 0))

    def _post_checkpoint_reached(self):
        response = requests.post(
            self._param('checkpoint_reached_url'), json={}, timeout=self._param('http_timeout_s')
        )
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            payload = {}
        return response.status_code, payload

    def _wait_for_checkpoint(self, goal_handle, checkpoint: Checkpoint):
        threshold = float(self._param('proximity_threshold_m'))
        with self._gps_condition:
            while not goal_handle.is_cancel_requested:
                distance = self._distance_to(checkpoint)
                self._current_distance_m = distance
                if distance <= threshold:
                    return True
                self._publish_feedback(goal_handle, checkpoint.sequence, distance, 'navigating')
                self._gps_condition.wait(timeout=0.5)
        return False

    def _publish_feedback(self, goal_handle, sequence: int, distance: float, state: str):
        feedback = StartMission.Feedback()
        feedback.current_checkpoint_sequence = sequence
        feedback.distance_m = distance
        feedback.state = state
        goal_handle.publish_feedback(feedback)

    def _verification_loop(self, goal_handle, checkpoint: Checkpoint):
        while not goal_handle.is_cancel_requested:
            status, payload = self._post_checkpoint_reached()
            if status == 200:
                return True, bool(payload.get('mission_completed', False)), payload
            if status == 400:
                self.get_logger().warn('Checkpoint endpoint rejected the checkpoint; resuming approach.')
                self._resume_motion()
                return False, False, payload
            if status == 503:
                self.get_logger().warn('Final stop was not confirmed; holding zero and retrying.')
                self._request_stop()
                self._publish_feedback(goal_handle, checkpoint.sequence, 0.0, 'confirming_stop')
                with self._gps_condition:
                    self._gps_condition.wait(timeout=float(self._param('verification_retry_s')))
                continue
            raise RuntimeError(f'checkpoint-reached returned HTTP {status}')
        return False, False, {}

    def _execute_callback(self, goal_handle):
        result = StartMission.Result()
        try:
            checkpoints, latest_scanned = self._fetch_checkpoints()
            start_sequence = latest_scanned + 1 if goal_handle.request.resume_from_latest_scanned else 1
            checkpoint_index = next(
                (index for index, item in enumerate(checkpoints) if item.sequence >= start_sequence),
                len(checkpoints),
            )
            if checkpoint_index >= len(checkpoints):
                goal_handle.succeed()
                result.success = True
                result.mission_completed = True
                result.last_checkpoint_sequence = latest_scanned
                result.message = 'Mission already completed according to checkpoint-list.'
                return result

            with self._lock:
                self._mission_active = True
                self._motion_allowed = False
                self._stop_requested = True

            while checkpoint_index < len(checkpoints):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.message = 'Mission canceled.'
                    return result

                checkpoint = checkpoints[checkpoint_index]
                with self._lock:
                    start_lat = self._current_lat if self._current_lat is not None else checkpoint.latitude
                    start_lon = self._current_lon if self._current_lon is not None else checkpoint.longitude
                
                waypoints = self._generate_waypoints(start_lat, start_lon, checkpoint.latitude, checkpoint.longitude)
                self.get_logger().info(
                    f'Navigating to checkpoint sequence {checkpoint.sequence} '
                    f'({checkpoint.latitude:.8f}, {checkpoint.longitude:.8f}) via {len(waypoints)} waypoints.'
                )

                # Navigate through each waypoint
                for waypoint_idx, (wp_lat, wp_lon) in enumerate(waypoints):
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        result.message = 'Mission canceled.'
                        return result

                    # Create temporary checkpoint for waypoint
                    waypoint = Checkpoint(
                        checkpoint_id=checkpoint.checkpoint_id,
                        sequence=checkpoint.sequence,
                        latitude=wp_lat,
                        longitude=wp_lon,
                    )
                    progress = int((waypoint_idx + 1) * 10)  # 10, 20, ..., 100
                    
                    self._publish_goal(waypoint)
                    if waypoint_idx == 0:
                        self.get_logger().info(f'  Waypoint {progress}%: lat={wp_lat:.8f}, lon={wp_lon:.8f}')
                    
                    if not self._wait_for_checkpoint(goal_handle, waypoint):
                        goal_handle.canceled()
                        result.message = 'Mission canceled.'
                        return result
                    
                    if waypoint_idx < len(waypoints) - 1:
                        self.get_logger().info(f'  Reached {progress}%, moving to next waypoint')

                # We've reached the final waypoint (checkpoint itself)
                self._request_stop()
                self._publish_feedback(goal_handle, checkpoint.sequence, 0.0, 'confirming_checkpoint')
                reached, mission_completed, payload = self._verification_loop(goal_handle, checkpoint)
                if not reached:
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        result.message = 'Mission canceled.'
                        return result
                    continue

                result.last_checkpoint_sequence = checkpoint.sequence
                if mission_completed:
                    goal_handle.succeed()
                    result.success = True
                    result.mission_completed = True
                    result.message = payload.get('message', 'Mission completed.')
                    return result

                next_sequence = int(payload.get('next_checkpoint_sequence', checkpoint.sequence + 1))
                checkpoint_index = next(
                    (index for index, item in enumerate(checkpoints) if item.sequence >= next_sequence),
                    len(checkpoints),
                )

            goal_handle.succeed()
            result.success = True
            result.mission_completed = True
            result.message = 'All checkpoints processed.'
            return result
        except Exception as exc:
            self.get_logger().error(f'Mission failed: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = str(exc)
            return result
        finally:
            self._enable_pub.publish(Bool(data=False))
            with self._lock:
                self._mission_active = False
                self._motion_allowed = False
                self._stop_requested = True
            self._publish_zero()

    def destroy_node(self):
        self._action_server.destroy()
        self._publish_zero()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CheckpointControllerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
