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

The planned path is reduced to a route by line-of-sight shortcutting against
the same costmap it was planned over, so every straight segment of the route is
guaranteed free of lethal cells (keeping 10 evenly indexed poses instead put
27-47 lethal cells back under the segments in three test scenarios -- see
test/test_waypoint_reduction.py).

The route is driven with a carrot: a goal kept a fixed distance ahead of the
rover's projection onto the route and re-published several times a second,
rather than a few waypoints each held until the rover enters a proximity
radius. Two things measured on mission_10sept forced that:
- The model reads the goal as a displacement and was trained on goals 0-6 s
  ahead (p50 1.6 m, p99 5.2 m). Waypoints up to 15 m apart, released at 8 m,
  put the goal 9-20 m away on 100% of ticks.
- Shrinking spacing and radius does not fix that: the rover ran 3-6 m off the
  route, so a small radius is never entered and the goal falls behind (one
  waypoint switch in the whole run at 3 m / 1.5 m). A carrot has no "reached"
  state to miss; replayed on the same trajectory it kept the goal inside the
  training range on 76% of ticks at 1.5 m ahead.
"""

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import requests

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32

from erc_inference_msgs.action import StartMission
from erc_static_map_msgs.srv import GenerateCostmap, PlanPath


# nav_msgs/OccupancyGrid convention: 0-100 is a probability of occupancy and
# -1 is unknown, so >= 100 means "definitely blocked". erc_static_map_node
# writes exactly 100 for building footprints (its LETHAL_COST) and leaves
# unmapped-but-traversable terrain at 0; unknown (-1) is never emitted, and
# would be treated as traversable here, which matches that node's stated
# "unmapped, open, traversable" reading.
LETHAL_COST = 100


# The goal and modality topics carry LATCHED STATE, not events: "the goal is
# here", "read the pose token, ignore the satellite one". The goal is
# re-published several times a second but the modality flags only once per
# leg, so with the default volatile QoS an inference node that starts (or
# restarts) mid-leg receives no modality and silently falls
# back to its own defaults -- which select modality 0 (satellite), where the
# model masks the GPS goal out entirely. The rover then drives on visual habit
# with no goal and nothing reports it.
#
# TRANSIENT_LOCAL with depth 1 makes a late joiner receive the current value
# immediately. Both ends must declare it: a volatile publisher and a
# transient-local subscriber are INCOMPATIBLE and never connect at all, which
# is why omnivla_edge_node.py and prueba.py carry the same profile. It is also
# why manual `ros2 topic pub` on these topics now needs
# `--qos-durability transient_local` (see omnivla_edge_node.py's header).
LATCHED_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


# How far past the rover's last route projection the next one may be found.
# A route that doubles back close to itself would otherwise let the
# projection jump onto its later stretch and skip everything in between.
PROJECTION_WINDOW_M = 10.0
_METERS_PER_DEG_LAT = math.radians(1.0) * 6378137.0


def latlon_to_local(frame, lat, lon):
    lat0, lon0, k_east, k_north = frame
    return (lon - lon0) * k_east, (lat - lat0) * k_north


def local_to_latlon(frame, east, north):
    lat0, lon0, k_east, k_north = frame
    return lat0 + north / k_north, lon0 + east / k_east


def route_to_local(route):
    """[(lat, lon), ...] -> (frame, points (N, 2) east/north m, cumulative arc length).

    Equirectangular about the first point: over one leg (tens of meters) its
    error is millimetric, and it needs no UTM zone.
    """
    lat0, lon0 = route[0]
    frame = (lat0, lon0, _METERS_PER_DEG_LAT * math.cos(math.radians(lat0)), _METERS_PER_DEG_LAT)
    pts = np.array([latlon_to_local(frame, lat, lon) for lat, lon in route], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    return frame, pts, cum


def project_forward(pts, cum, east, north, s_min, window_m=PROJECTION_WINDOW_M):
    """Arc length of the route point closest to (east, north), never below s_min.

    Only segments overlapping [s_min, s_min + window_m] are searched, and the
    result never moves backward: the carrot has to keep leading the rover.
    """
    best_s, best_d = s_min, math.inf
    for i in range(len(pts) - 1):
        if cum[i + 1] < s_min:
            continue
        if cum[i] > s_min + window_m:
            break
        a = pts[i]
        ab = pts[i + 1] - a
        seg2 = float(ab @ ab)
        t = 0.0 if seg2 == 0.0 else min(1.0, max(0.0, ((east - a[0]) * ab[0] + (north - a[1]) * ab[1]) / seg2))
        d = math.hypot(east - (a[0] + t * ab[0]), north - (a[1] + t * ab[1]))
        if d < best_d:
            best_d, best_s = d, cum[i] + t * math.sqrt(seg2)
    return max(s_min, best_s)


def point_at(pts, cum, s):
    """(east, north, route bearing in degrees, 0 = North, clockwise) at arc length s.

    s is clamped to the route, so a carrot running off the end sits on the
    final point and keeps the last segment's bearing.
    """
    s = min(max(s, 0.0), float(cum[-1]))
    i = int(min(max(np.searchsorted(cum, s, side='right') - 1, 0), len(pts) - 2))
    a, b = pts[i], pts[i + 1]
    seg = cum[i + 1] - cum[i]
    t = 0.0 if seg <= 0.0 else (s - cum[i]) / seg
    bearing = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0
    return float(a[0] + t * (b[0] - a[0])), float(a[1] + t * (b[1] - a[1])), bearing


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
        # Position for leg starts, carrot projection and arrival: the EKF-filtered
        # fix (navsat_transform, 10 Hz) rather than the raw SDK fix, which
        # refreshes every ~1.3 s in 0.4 m steps. Needs erc_localization's
        # localization_global.launch.py running.
        self.declare_parameter('gps_topic', '/erc/gps/filtered')
        self.declare_parameter('model_cmd_vel_topic', '/omnivla/cmd_vel')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('goal_gps_topic', '/goal_gps')
        self.declare_parameter('goal_compass_topic', '/goal_compass')
        self.declare_parameter('use_pose_goal_topic', '/use_pose_goal')
        self.declare_parameter('use_satellite_topic', '/use_satellite')
        self.declare_parameter('use_image_goal_topic', '/use_image_goal')
        self.declare_parameter('use_lan_prompt_topic', '/use_lan_prompt')
        self.declare_parameter('enable_inference_topic', '/enable_inference')
        # Arrival radius for a checkpoint. The SDK decides for itself whether a
        # checkpoint counts -- its radius is not documented anywhere in this
        # repo -- and the old 8 m had the rover declare arrival at 7.7 m on
        # mission_10sept. After a rejection the radius halves, down to
        # min_checkpoint_proximity_m, so the rover closes in instead of
        # re-posting from the same spot.
        self.declare_parameter('checkpoint_proximity_m', 3.0)
        self.declare_parameter('min_checkpoint_proximity_m', 1.0)
        # How far ahead along the route the carrot sits, and how often it is
        # re-published. 1.5 m kept the goal inside the model's training range
        # (<= 5.2 m) on 76% of mission_10sept's ticks (2.0 m: 71%, 3.0 m: 60%);
        # the goal distance is sqrt(lookahead^2 + cross-track^2), so it is never
        # shorter than this. 3 Hz matches the inference tick.
        self.declare_parameter('carrot_distance_m', 1.5)
        self.declare_parameter('carrot_rate_hz', 3.0)
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
        # Longest straight segment of a route. Line-of-sight shortcutting
        # already guarantees each segment is obstacle-free; the cap bounds the
        # shortcut search and how long a single straight run can get. It no
        # longer sets how far away the goal is -- the carrot does.
        self.declare_parameter('max_waypoint_spacing_m', 15.0)

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

        # cmd_vel is streaming data and stays volatile; everything below it is
        # latched state (see LATCHED_QOS).
        self._cmd_pub = self.create_publisher(Twist, self._param('cmd_vel_topic'), 10)
        self._goal_gps_pub = self.create_publisher(NavSatFix, self._param('goal_gps_topic'), LATCHED_QOS)
        self._goal_compass_pub = self.create_publisher(Float32, self._param('goal_compass_topic'), LATCHED_QOS)
        self._use_pose_pub = self.create_publisher(Bool, self._param('use_pose_goal_topic'), LATCHED_QOS)
        self._use_satellite_pub = self.create_publisher(Bool, self._param('use_satellite_topic'), LATCHED_QOS)
        self._use_image_pub = self.create_publisher(Bool, self._param('use_image_goal_topic'), LATCHED_QOS)
        self._use_lan_pub = self.create_publisher(Bool, self._param('use_lan_prompt_topic'), LATCHED_QOS)
        self._enable_pub = self.create_publisher(Bool, self._param('enable_inference_topic'), LATCHED_QOS)

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

    def _publish_modality(self):
        """Select modality 4 (pose goal only) on the inference node.

        pose=True with the other three False is what compute_modality_id maps to
        4, the GPS-only modality: the model reads the goal_pose token and masks
        the satellite, goal-image and language ones. Satellite is False because
        no satellite tile is ever supplied -- the inference node feeds a black
        placeholder -- so any modality that reads that token navigates blind.
        """
        self._use_pose_pub.publish(Bool(data=True))
        self._use_satellite_pub.publish(Bool(data=False))
        self._use_image_pub.publish(Bool(data=False))
        self._use_lan_pub.publish(Bool(data=False))

    def _publish_carrot(self, lat: float, lon: float, bearing_deg: float):
        goal = NavSatFix()
        goal.latitude = lat
        goal.longitude = lon
        self._goal_gps_pub.publish(goal)
        # The heading the rover should have at the carrot: the route's own
        # direction there, in the SDK compass convention (0 = North,
        # clockwise-positive) that /erc/heading_deg uses. The model reads it as
        # cos/sin(goal - current heading); in its training data that difference
        # was within 20 deg for 88% of samples, because it was the heading the
        # robot really had on arrival. The constant 0.0 sent before meant
        # "arrive facing north" on every leg and so fed the model the rover's
        # absolute heading instead.
        self._goal_compass_pub.publish(Float32(data=float(bearing_deg)))

    def _start_motion(self, sequence: int):
        self._publish_modality()
        self._enable_pub.publish(Bool(data=True))
        with self._lock:
            self._motion_allowed = True
            self._stop_requested = False
            self._current_sequence = sequence

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
        # Re-assert the modality, not just the enable flag: if the inference
        # node restarted during the pause it would otherwise come back on its
        # own defaults (modality 0, satellite). LATCHED_QOS already covers the
        # restart case on its own; this is the belt to that pair of braces, and
        # it costs four Bool publishes on a path that runs once per checkpoint.
        self._publish_modality()
        self._enable_pub.publish(Bool(data=True))

    def _publish_zero(self):
        self._cmd_pub.publish(Twist())

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

    def _generate_route(self, start_lat: float, start_lon: float,
                        target_lat: float, target_lon: float) -> List[Tuple[float, float]]:
        """Route for one leg (start -> target) as [(lat, lon), ...], start included.

        Calls erc_static_map/generate_costmap for this leg's own A (start)
        and B (target), then erc_static_map/plan_path with the response --
        passing utm_crs/origin_utm straight through, so this node never
        guesses or separately derives either one (see module docstring).
        Falls back to the straight line start -> target if either service is
        unavailable, times out, or fails, or if planning finds no path.
        """
        log = self.get_logger()
        straight = [(start_lat, start_lon), (target_lat, target_lon)]

        costmap_req = GenerateCostmap.Request()
        costmap_req.origin_lat = start_lat
        costmap_req.origin_lon = start_lon
        costmap_req.checkpoint_lat = target_lat
        costmap_req.checkpoint_lon = target_lon

        costmap_resp = self._call_service_sync(
            self._costmap_client, costmap_req, float(self._param('costmap_service_timeout_s'))
        )
        if costmap_resp is None:
            log.warn('generate_costmap service unavailable or timed out; driving a straight line')
            return straight
        if not costmap_resp.success:
            log.warn(f'generate_costmap failed ({costmap_resp.message}); driving a straight line')
            return straight

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
            log.warn('plan_path service unavailable or timed out; driving a straight line')
            return straight
        if not plan_resp.success or len(plan_resp.path.poses) < 2:
            log.warn(f'plan_path failed ({plan_resp.message}); driving a straight line')
            return straight

        route = self._path_to_route(
            plan_resp.path, costmap_resp.utm_crs,
            (costmap_resp.origin_utm_x, costmap_resp.origin_utm_y),
            costmap=costmap_resp.costmap,
        )
        return route if len(route) >= 2 else straight

    @staticmethod
    def _grid_occupancy(grid):
        """(occupied[row, col], resolution, origin_x, origin_y) from an OccupancyGrid."""
        h, w = grid.info.height, grid.info.width
        if h == 0 or w == 0 or grid.info.resolution <= 0.0:
            return None
        data = np.asarray(grid.data, dtype=np.int16).reshape((h, w))
        return (data >= LETHAL_COST,
                float(grid.info.resolution),
                float(grid.info.origin.position.x),
                float(grid.info.origin.position.y))

    @staticmethod
    def _segment_is_clear(occ, res, ox, oy, p, q, step_ratio=0.4):
        """Is the straight segment p->q free of lethal cells?

        Sampled at 0.4 of a cell, which cannot skip over a cell whose width is
        one full cell. A leaving-the-grid segment counts as blocked: off-map is
        exactly the terrain nothing has checked.
        """
        h, w = occ.shape
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        n = max(1, int(math.ceil(length / (step_ratio * res))))
        for i in range(n + 1):
            t = i / n
            c = int(math.floor((p[0] + dx * t - ox) / res))
            r = int(math.floor((p[1] + dy * t - oy) / res))
            if not (0 <= c < w and 0 <= r < h) or occ[r, c]:
                return False
        return True

    @classmethod
    def _shortcut_path(cls, points, grid, max_spacing_m):
        """Greedy line-of-sight shortcutting: the fewest waypoints such that every
        straight segment between consecutive ones is obstacle-free.

        This replaces "keep 10 evenly indexed poses", which discarded exactly the
        detail the planner existed to produce -- every detour finer than the
        sampling interval became a straight line nobody had checked.

        Look-ahead is capped at max_spacing_m, which both bounds the segment
        length and keeps this O(n * cap) instead of O(n^2) on a long open run.
        """
        occupancy = cls._grid_occupancy(grid)
        if occupancy is None or len(points) < 2:
            return list(points)
        occ, res, ox, oy = occupancy

        out = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            best = anchor + 1
            for j in range(anchor + 1, len(points)):
                if math.dist(points[anchor], points[j]) > max_spacing_m:
                    break
                # Stop extending at the first blocked segment rather than
                # scanning past it: a farther point being visible again does not
                # make the segment through the obstacle safe.
                if not cls._segment_is_clear(occ, res, ox, oy, points[anchor], points[j]):
                    break
                best = j
            out.append(points[best])
            anchor = best
        return out

    @staticmethod
    def _resample_by_distance(points, spacing_m):
        """Fallback when no usable costmap is available: keep the endpoints and one
        point roughly every spacing_m. Still follows the planned path -- unlike a
        fixed count, the sampling interval does not grow with the leg length."""
        if len(points) < 2:
            return list(points)
        out = [points[0]]
        for pt in points[1:]:
            if math.dist(out[-1], pt) >= spacing_m:
                out.append(pt)
        if out[-1] != points[-1]:
            out.append(points[-1])
        return out

    def _path_to_route(self, path, utm_crs: str, origin_utm: Tuple[float, float],
                       costmap=None) -> List[Tuple[float, float]]:
        """Reduce a planned Path to a GPS route, preserving obstacle avoidance.

        utm_crs / origin_utm are the SAME values erc_static_map_node used to
        build the costmap this path was planned over (passed straight
        through from the GenerateCostmap response -- see _generate_route),
        so this is a direct inverse of that node's local-ENU -> UTM -> GPS
        chain rather than a guess. The first point, where the rover stood when
        the leg was planned, is kept: the carrot needs the first segment.
        """
        import pyproj

        poses = path.poses
        if len(poses) < 2:
            return []

        max_spacing = float(self._param('max_waypoint_spacing_m'))
        points = [(ps.pose.position.x, ps.pose.position.y) for ps in poses]

        if costmap is not None:
            kept = self._shortcut_path(points, costmap, max_spacing)
            self.get_logger().info(
                f'Path reduced {len(points)} -> {len(kept)} route points by line-of-sight '
                f'shortcutting (max spacing {max_spacing:.0f} m)')
        else:
            kept = self._resample_by_distance(points, max_spacing)
            self.get_logger().warn(
                f'No costmap available to verify route segments; falling back to '
                f'distance resampling ({len(points)} -> {len(kept)} points). Straight '
                f'lines between these are NOT checked against obstacles.')

        origin_x, origin_y = origin_utm
        transformer = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

        route = []
        for x, y in kept:
            lon, lat = transformer.transform(x + origin_x, y + origin_y)
            route.append((lat, lon))
        return route

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

    def _follow_route(self, goal_handle, checkpoint: Checkpoint, route, arrival_threshold_m: float) -> bool:
        """Drive `route` by streaming a carrot to the inference node.

        The carrot is the route point carrot_distance_m of arc length ahead of
        the rover's projection onto the route, and its heading is the route's
        own direction there. Returns True once the rover is within
        arrival_threshold_m of the checkpoint, False if the mission is canceled.
        """
        frame, pts, cum = route_to_local(route)
        goal_e, goal_n = latlon_to_local(frame, checkpoint.latitude, checkpoint.longitude)
        lookahead = float(self._param('carrot_distance_m'))
        period = 1.0 / float(self._param('carrot_rate_hz'))
        total = float(cum[-1])
        s_proj = 0.0
        started = False
        last_publish = 0.0
        reported_quarter = 0
        with self._gps_condition:
            while not goal_handle.is_cancel_requested:
                if self._current_lat is not None and self._current_lon is not None:
                    east, north = latlon_to_local(frame, self._current_lat, self._current_lon)
                    remaining = math.hypot(goal_e - east, goal_n - north)
                    self._current_distance_m = remaining
                    if remaining <= arrival_threshold_m:
                        return True
                    s_proj = project_forward(pts, cum, east, north, s_proj)
                    now = time.monotonic()
                    if now - last_publish >= period:
                        last_publish = now
                        carrot_e, carrot_n, carrot_bearing = point_at(pts, cum, s_proj + lookahead)
                        self._publish_carrot(*local_to_latlon(frame, carrot_e, carrot_n), carrot_bearing)
                        if not started:
                            # Only once a goal is out: the inference node will
                            # not drive a pose goal it has not received.
                            self._start_motion(checkpoint.sequence)
                            started = True
                        self._publish_feedback(goal_handle, checkpoint.sequence, remaining, 'navigating')
                    quarter = int(4 * s_proj / total) if total > 0.0 else 0
                    if quarter > reported_quarter:
                        reported_quarter = quarter
                        self.get_logger().info(
                            f'  {100 * s_proj / total:.0f}% of the route ({s_proj:.1f}/{total:.1f} m), '
                            f'{remaining:.1f} m to the checkpoint')
                self._gps_condition.wait(timeout=period)
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

            arrival_for_sequence = None
            arrival_threshold_m = float(self._param('checkpoint_proximity_m'))
            while checkpoint_index < len(checkpoints):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.message = 'Mission canceled.'
                    return result

                checkpoint = checkpoints[checkpoint_index]
                with self._lock:
                    start_lat = self._current_lat if self._current_lat is not None else checkpoint.latitude
                    start_lon = self._current_lon if self._current_lon is not None else checkpoint.longitude
                
                if arrival_for_sequence != checkpoint.sequence:
                    arrival_for_sequence = checkpoint.sequence
                    arrival_threshold_m = float(self._param('checkpoint_proximity_m'))

                route = self._generate_route(start_lat, start_lon, checkpoint.latitude, checkpoint.longitude)
                self.get_logger().info(
                    f'Navigating to checkpoint sequence {checkpoint.sequence} '
                    f'({checkpoint.latitude:.8f}, {checkpoint.longitude:.8f}) along a {len(route)}-point '
                    f'route, carrot {float(self._param("carrot_distance_m")):.1f} m ahead, '
                    f'arrival within {arrival_threshold_m:.1f} m.'
                )
                if not self._follow_route(goal_handle, checkpoint, route, arrival_threshold_m):
                    goal_handle.canceled()
                    result.message = 'Mission canceled.'
                    return result

                self._request_stop()
                self._publish_feedback(goal_handle, checkpoint.sequence, self._current_distance_m,
                                       'confirming_checkpoint')
                self.get_logger().info(
                    f'Within {arrival_threshold_m:.1f} m of checkpoint {checkpoint.sequence} '
                    f'({self._current_distance_m:.1f} m); asking the SDK to confirm.')
                reached, mission_completed, payload = self._verification_loop(goal_handle, checkpoint)
                if not reached:
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        result.message = 'Mission canceled.'
                        return result
                    # Rejected, so the SDK's radius is tighter than ours. The
                    # rover is already inside our radius, so re-approaching with
                    # it would re-post from the same spot forever: close in first.
                    tighter = max(float(self._param('min_checkpoint_proximity_m')), 0.5 * arrival_threshold_m)
                    self.get_logger().warn(
                        f'Checkpoint {checkpoint.sequence} rejected at {self._current_distance_m:.1f} m; '
                        f'approaching again until within {tighter:.1f} m.')
                    arrival_threshold_m = tighter
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
