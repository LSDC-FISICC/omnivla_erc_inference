#!/usr/bin/env python3
"""High-level checkpoint mission controller for the OmniVLA edge node.

Waypoints are generated using A* path planning on a static costmap if the
costmap/planner services are available, falling back to linear GPS
interpolation otherwise. This avoids untraversable terrain (buildings,
obstacles) when planning routes to checkpoints.

Costmap generation and A* planning are now ROS services
(erc_static_map_msgs/srv/GenerateCostmap and .../PlanPath), served by
erc_static_map_node and erc_astar_planner_node respectively, instead of a
one-shot topic/parameter pipeline. This node calls GenerateCostmap once per
leg with that leg's own start/goal GPS, then PlanPath with the response --
so a multi-checkpoint mission gets a correctly-scoped costmap per leg
without needing a separate process per leg, and without ever having to
guess or separately latch a UTM zone/origin: GenerateCostmap's response
carries utm_crs/origin_utm_x/origin_utm_y and those are passed straight
through to PlanPath.
"""

import json
import math
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple

import requests

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32

from erc_inference_msgs.action import StartMission
from erc_static_map_msgs.srv import GenerateCostmap, PlanPath


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: int
    sequence: int
    latitude: float
    longitude: float


class CheckpointControllerNode(Node):

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
        # Costmap/planner service call tuning. Costmap generation involves a
        # network-bound OSM query and can be slow; give it a generous
        # timeout and treat a timeout/failure the same as "service
        # unavailable" -- fall back to linear interpolation rather than
        # blocking the mission indefinitely on one leg's map.
        self.declare_parameter('costmap_service_timeout_s', 60.0)
        self.declare_parameter('planner_service_timeout_s', 15.0)

        self._lock = threading.RLock()
        self._gps_condition = threading.Condition(self._lock)
        self._current_lat: Optional[float] = None
        self._current_lon: Optional[float] = None
        self._last_model_cmd = Twist()
        self._mission_active = False
        self._motion_allowed = False
        self._stop_requested = True
        self._current_sequence = 0
        self._current_distance_m = float('inf')

        self._model_cmd_sub = self.create_subscription(
            Twist, self._param('model_cmd_vel_topic'), self._model_cmd_callback, 10
        )
        self._gps_sub = self.create_subscription(
            NavSatFix, self._param('gps_topic'), self._gps_callback, 10
        )

        self._cmd_pub = self.create_publisher(Twist, self._param('cmd_vel_topic'), 10)
        self._goal_gps_pub = self.create_publisher(NavSatFix, self._param('goal_gps_topic'), 10)
        self._goal_compass_pub = self.create_publisher(Float32, self._param('goal_compass_topic'), 10)
        self._use_pose_pub = self.create_publisher(Bool, self._param('use_pose_goal_topic'), 10)
        self._use_satellite_pub = self.create_publisher(Bool, self._param('use_satellite_topic'), 10)
        self._use_image_pub = self.create_publisher(Bool, self._param('use_image_goal_topic'), 10)
        self._use_lan_pub = self.create_publisher(Bool, self._param('use_lan_prompt_topic'), 10)
        self._enable_pub = self.create_publisher(Bool, self._param('enable_inference_topic'), 10)

        self._costmap_client = self.create_client(GenerateCostmap, 'erc_static_map/generate_costmap')
        self._planner_client = self.create_client(PlanPath, 'erc_static_map/plan_path')

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
            self._gps_condition.notify_all()

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

    def _call_service_sync(self, client, request, timeout_s: float):
        """Call a service and block the calling thread until it completes or
        times out. _execute_callback runs on a MultiThreadedExecutor worker
        thread (not the main spin thread), so blocking here is fine -- it
        does not stall other callbacks, unlike call_async().result() from a
        single-threaded context."""
        if not client.wait_for_service(timeout_sec=timeout_s):
            return None
        future = client.call_async(request)
        event = threading.Event()
        future.add_done_callback(lambda _f: event.set())
        if not event.wait(timeout=timeout_s):
            return None
        try:
            return future.result()
        except Exception as e:
            self.get_logger().warn(f'Service call raised: {e}')
            return None

    def _generate_waypoints(self, start_lat: float, start_lon: float, target_lat: float, target_lon: float) -> list:
        """Generate GPS waypoints for one leg (start -> target).

        Calls erc_static_map/generate_costmap for this leg's own A (start)
        and B (target), then erc_static_map/plan_path with the response --
        passing utm_crs/origin_utm straight through, so this node never
        guesses or separately derives either one (see module docstring).
        Falls back to linear GPS interpolation if either service is
        unavailable, times out, or fails, or if planning finds no path.
        """
        log = self.get_logger()

        costmap_req = GenerateCostmap.Request()
        costmap_req.origin_lat = start_lat
        costmap_req.origin_lon = start_lon
        costmap_req.checkpoint_lat = target_lat
        costmap_req.checkpoint_lon = target_lon

        costmap_resp = self._call_service_sync(
            self._costmap_client, costmap_req, float(self._param('costmap_service_timeout_s'))
        )
        if costmap_resp is None:
            log.warn('generate_costmap service unavailable or timed out; using linear interpolation')
            return self._linear_waypoints(start_lat, start_lon, target_lat, target_lon)
        if not costmap_resp.success:
            log.warn(f'generate_costmap failed ({costmap_resp.message}); using linear interpolation')
            return self._linear_waypoints(start_lat, start_lon, target_lat, target_lon)

        plan_req = PlanPath.Request()
        plan_req.origin_lat = start_lat
        plan_req.origin_lon = start_lon
        plan_req.checkpoint_lat = target_lat
        plan_req.checkpoint_lon = target_lon
        plan_req.costmap = costmap_resp.costmap
        plan_req.utm_crs = costmap_resp.utm_crs
        plan_req.origin_utm_x = costmap_resp.origin_utm_x
        plan_req.origin_utm_y = costmap_resp.origin_utm_y

        plan_resp = self._call_service_sync(
            self._planner_client, plan_req, float(self._param('planner_service_timeout_s'))
        )
        if plan_resp is None:
            log.warn('plan_path service unavailable or timed out; using linear interpolation')
            return self._linear_waypoints(start_lat, start_lon, target_lat, target_lon)
        if not plan_resp.success or len(plan_resp.path.poses) < 2:
            log.warn(f'plan_path failed ({plan_resp.message}); using linear interpolation')
            return self._linear_waypoints(start_lat, start_lon, target_lat, target_lon)

        waypoints = self._path_to_waypoints(
            plan_resp.path, costmap_resp.utm_crs,
            (costmap_resp.origin_utm_x, costmap_resp.origin_utm_y),
        )
        if waypoints:
            log.info(f'Generated {len(waypoints)} A* waypoints to avoid obstacles')
            return waypoints
        return self._linear_waypoints(start_lat, start_lon, target_lat, target_lon)

    @staticmethod
    def _path_to_waypoints(path, utm_crs: str, origin_utm: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Sample 10 waypoints evenly along a planned Path and convert each
        pose (local-ENU, relative to origin_utm) back to GPS.

        utm_crs / origin_utm are the SAME values erc_static_map_node used to
        build the costmap this path was planned over (passed straight
        through from the GenerateCostmap response -- see _generate_waypoints),
        so this is a direct inverse of that node's local-ENU -> UTM -> GPS
        chain rather than a guess.
        """
        import pyproj

        poses = path.poses
        if len(poses) < 2:
            return []

        origin_x, origin_y = origin_utm
        transformer = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

        def local_to_gps(x: float, y: float) -> Tuple[float, float]:
            lon, lat = transformer.transform(x + origin_x, y + origin_y)
            return lat, lon

        waypoints = []
        for i in range(1, 11):
            idx = min(int(len(poses) * i / 10.0), len(poses) - 1)
            pos = poses[idx].pose.position
            waypoints.append(local_to_gps(pos.x, pos.y))
        return waypoints

    def _linear_waypoints(self, start_lat: float, start_lon: float, target_lat: float, target_lon: float) -> list:
        """Fallback: 10 GPS waypoints at even fractions of the straight-line
        geodesic (equirectangular approximation) from start to target."""
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
