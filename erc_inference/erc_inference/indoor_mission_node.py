#!/usr/bin/env python3
"""Checkpoint missions without GPS: checkpoint_controller_node for indoor tracks.

Indoors there is no fix, so none of the outdoor chain works: the inference
nodes wait for /erc/gps/filtered before running the model, localization_global
blocks on the first fix, and the A* costmap comes from OSM. This node replaces
the outdoor chain's localization and route source and leaves everything else
alone:

- Position and heading: the local EKF (/erc/odometry/local: wheel speed + gyro
  yaw, no GPS, no magnetometer -- magnetometers are unreliable next to steel
  and wiring) re-anchored to the track frame of `route_file` at mission start.
  See indoor_track.py for the frame and why the datum is where it is.
- The inference node is fed that pose as a pseudo-fix on `fix_topic` and a
  compass heading on `heading_topic`. mission_indoor.launch.py points the
  model node's gps_topic/compass_topic at them, so omnivla_edge_node and
  omnivla_original_node run unmodified, with the same controller.yaml.
- Route: the checkpoints and corridor corners in `route_file`, driven with the
  same carrot as outdoors (checkpoint_controller_node.project_forward /
  point_at), re-published as /goal_gps + /goal_compass with modality 4 (pose
  goal), or 8 (pose + language) with use_language:=true.

Unchanged from checkpoint_controller_node: the StartMission action, this node
being the only writer of /cmd_vel (kill it to take the rover back), the
CommandShaper acceleration limits, and the /checkpoint-reached confirmation
loop (confirm_with_sdk:=false accepts arrival locally, for rehearsals).

Dead reckoning drifts without bound, and nothing here corrects it: over the
132 m NYU loop a 2% wheel-scale or a few-degree yaw error is meters, against
corridors ~2-2.5 m wide. What keeps the rover off the walls is the model's own
corridor following (polar.steering_source plan) -- see test/indoor_sim.py for
how much drift the carrot alone tolerates, and docs/PLAN_INDOOR_NYU.md for the
corrections planned (cones, corners).

    ros2 run erc_inference indoor_mission_node
    ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \
        "{resume_from_latest_scanned: false}"

resume_from_latest_scanned: true re-anchors the rover ON the last confirmed
checkpoint, facing the next leg. Put it there before sending the goal.
"""

import json
import math
import os
import threading
import time
from typing import Optional

import numpy as np
import requests

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Bool, Float32, String

from erc_inference_msgs.action import StartMission

from erc_inference.checkpoint_controller_node import LATCHED_QOS, point_at, project_forward
from erc_inference.indoor_track import (IndoorCheckpoint, Pose2D, anchor, load_track,
                                        track_to_latlon, yaw_to_compass_deg)
from erc_inference.motion_control import CommandShaper

TRACK_FRAME = 'track'


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class IndoorMissionNode(Node):

    def __init__(self, node_name='indoor_mission_node', route_file='indoor_nyu_track.yaml'):
        super().__init__(node_name)

        self.declare_parameter('route_file', os.path.join(
            get_package_share_directory('erc_inference'), 'config', route_file))
        self.declare_parameter('odom_topic', '/erc/odometry/local')
        # Must match mission_indoor.launch.py's remap of the model node.
        self.declare_parameter('fix_topic', '/erc/indoor/fix')
        self.declare_parameter('heading_topic', '/erc/indoor/heading_deg')
        self.declare_parameter('pose_topic', '/erc/indoor/pose')
        self.declare_parameter('checkpoint_list_url', 'http://localhost:8000/checkpoints-list')
        self.declare_parameter('checkpoint_reached_url', 'http://localhost:8000/checkpoint-reached')
        # How the SDK validates a checkpoint indoors is not in the NYU PDF. True
        # asks it exactly as outdoors; false accepts the dead-reckoned arrival.
        self.declare_parameter('confirm_with_sdk', True)
        self.declare_parameter('model_cmd_vel_topic', '/omnivla/cmd_vel')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('goal_gps_topic', '/goal_gps')
        self.declare_parameter('goal_compass_topic', '/goal_compass')
        self.declare_parameter('lan_prompt_topic', '/lan_prompt')
        self.declare_parameter('use_pose_goal_topic', '/use_pose_goal')
        self.declare_parameter('use_satellite_topic', '/use_satellite')
        self.declare_parameter('use_image_goal_topic', '/use_image_goal')
        self.declare_parameter('use_lan_prompt_topic', '/use_lan_prompt')
        self.declare_parameter('enable_inference_topic', '/enable_inference')
        # Modality 8 (pose + each checkpoint's `prompt`) instead of 4. Untested.
        self.declare_parameter('use_language', False)
        # Smaller than outdoors (8 m): the cones stand in 2-4 m corridors, and the
        # SDK's own indoor radius is unknown. Halves on rejection, as outdoors.
        self.declare_parameter('checkpoint_proximity_m', 2.0)
        self.declare_parameter('min_checkpoint_proximity_m', 0.5)
        # Same values and reasons as checkpoint_controller_node.
        self.declare_parameter('carrot_distance_m', 1.5)
        self.declare_parameter('carrot_rate_hz', 3.0)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('max_linear_accel', 0.3)
        self.declare_parameter('max_linear_decel', 0.6)
        self.declare_parameter('max_angular_accel', 0.6)
        self.declare_parameter('max_angular_decel', 1.2)
        self.declare_parameter('http_timeout_s', 15.0)
        self.declare_parameter('verification_retry_s', 1.0)

        self.track = load_track(self._param('route_file'))
        self.get_logger().info(
            f'Track {self.track.name!r}: ' + ', '.join(
                f'{c.sequence}:{c.name}({c.x:.1f},{c.y:.1f})' for c in self.track.checkpoints))

        self._lock = threading.RLock()
        self._odom_condition = threading.Condition(self._lock)
        self._odom_pose: Optional[Pose2D] = None
        self._track_from_odom: Optional[Pose2D] = None
        self._track_pose: Optional[Pose2D] = None
        self._last_model_cmd = Twist()
        self._mission_active = False
        self._motion_allowed = False
        self._stop_requested = True
        self._current_distance_m = float('inf')
        self._last_confirmed = 0
        self._shaper = CommandShaper(*self._shaper_limits())
        self._last_output_time = None

        # Own callback groups: with every callback in the node's default (mutually exclusive)
        # group, a slow one -- a perception callback waiting on a planning lock -- stalls
        # odometry and the /cmd_vel timer with it (run_e2e_images.sh, 28-sept).
        self._odom_group = MutuallyExclusiveCallbackGroup()
        self._output_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Odometry, self._param('odom_topic'), self._odom_callback, 50,
                                 callback_group=self._odom_group)
        self.create_subscription(Twist, self._param('model_cmd_vel_topic'), self._model_cmd_callback, 10)
        self._fix_pub = self.create_publisher(NavSatFix, self._param('fix_topic'), 10)
        self._heading_pub = self.create_publisher(Float32, self._param('heading_topic'), 10)
        self._pose_pub = self.create_publisher(PoseStamped, self._param('pose_topic'), 10)
        self._route_pub = self.create_publisher(Path, '/erc/global_route', LATCHED_QOS)
        self._carrot_pub = self.create_publisher(PointStamped, '/erc/carrot', 10)

        self._cmd_pub = self.create_publisher(Twist, self._param('cmd_vel_topic'), 10)
        self._goal_gps_pub = self.create_publisher(NavSatFix, self._param('goal_gps_topic'), LATCHED_QOS)
        self._goal_compass_pub = self.create_publisher(Float32, self._param('goal_compass_topic'), LATCHED_QOS)
        # Volatile: the model nodes subscribe to /lan_prompt with the default QoS.
        self._lan_prompt_pub = self.create_publisher(String, self._param('lan_prompt_topic'), 10)
        self._use_pose_pub = self.create_publisher(Bool, self._param('use_pose_goal_topic'), LATCHED_QOS)
        self._use_satellite_pub = self.create_publisher(Bool, self._param('use_satellite_topic'), LATCHED_QOS)
        self._use_image_pub = self.create_publisher(Bool, self._param('use_image_goal_topic'), LATCHED_QOS)
        self._use_lan_pub = self.create_publisher(Bool, self._param('use_lan_prompt_topic'), LATCHED_QOS)
        self._enable_pub = self.create_publisher(Bool, self._param('enable_inference_topic'), LATCHED_QOS)

        self._action_server = ActionServer(
            self, StartMission, 'start_mission',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
        )
        self._output_timer = self.create_timer(1.0 / self._param('control_rate_hz'), self._publish_output,
                                               callback_group=self._output_group)
        self.get_logger().info(
            f'Indoor mission node ready; pseudo-fix on {self._param("fix_topic")}, heading on '
            f'{self._param("heading_topic")}. Waiting for start_mission.')

    def _param(self, name):
        return self.get_parameter(name).value

    def _goal_callback(self, _request):
        with self._lock:
            return GoalResponse.REJECT if self._mission_active else GoalResponse.ACCEPT

    # ------------------------------------------------------------ localization

    def _odom_callback(self, msg: Odometry):
        p = msg.pose.pose
        odom = Pose2D(p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        with self._odom_condition:
            self._odom_pose = odom
            if self._track_from_odom is None:
                # Until a mission anchors it: wherever the EKF is now is the
                # track's start, so the model node has a fix to become ready on.
                self._track_from_odom = anchor(self.track.start, odom)
            self._track_pose = self._track_from_odom.compose(odom)
            pose = self._track_pose
            self._odom_condition.notify_all()
        self._publish_pose(pose, msg.header.stamp)

    def _anchor_at(self, track_pose: Pose2D) -> bool:
        with self._lock:
            if self._odom_pose is None:
                return False
            self._track_from_odom = anchor(track_pose, self._odom_pose)
            self._track_pose = track_pose
        self.get_logger().info(
            f'Anchored: rover at track ({track_pose.x:.2f}, {track_pose.y:.2f}) '
            f'yaw {math.degrees(track_pose.yaw):.1f} deg.')
        return True

    def _publish_pose(self, pose: Pose2D, stamp):
        lat, lon = track_to_latlon(pose.x, pose.y)
        fix = NavSatFix()
        fix.header.stamp, fix.header.frame_id = stamp, TRACK_FRAME
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.latitude, fix.longitude = lat, lon
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self._fix_pub.publish(fix)
        self._heading_pub.publish(Float32(data=float(yaw_to_compass_deg(pose.yaw))))
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = stamp, TRACK_FRAME
        ps.pose.position.x, ps.pose.position.y = pose.x, pose.y
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(pose.yaw / 2), math.cos(pose.yaw / 2)
        self._pose_pub.publish(ps)

    # ------------------------------------------------------------ output (same as outdoors)

    def _model_cmd_callback(self, msg: Twist):
        with self._lock:
            self._last_model_cmd = msg

    def _shaper_limits(self):
        return (float(self._param('max_linear_accel')), float(self._param('max_linear_decel')),
                float(self._param('max_angular_accel')), float(self._param('max_angular_decel')))

    def _publish_output(self):
        now = time.monotonic()
        with self._lock:
            target = self._last_model_cmd if (
                self._mission_active and self._motion_allowed and not self._stop_requested) else Twist()
            dt = 0.0 if self._last_output_time is None else now - self._last_output_time
            self._last_output_time = now
            self._shaper.configure(*self._shaper_limits())
            linear, angular = self._shaper.step(target.linear.x, target.angular.z, dt)
        out = Twist()
        out.linear.x, out.angular.z = linear, angular
        self._cmd_pub.publish(out)

    def _publish_zero(self):
        with self._lock:
            self._shaper.reset()
        self._cmd_pub.publish(Twist())

    def _publish_modality(self, checkpoint: IndoorCheckpoint):
        """Modality 4 (pose goal only), or 8 (pose + language) with use_language.

        Satellite and goal image stay off: no tile and no goal image exist here,
        and the model nodes feed black placeholders for both.
        """
        language = bool(self._param('use_language')) and bool(checkpoint.prompt)
        if language:
            self._lan_prompt_pub.publish(String(data=checkpoint.prompt))
        self._use_pose_pub.publish(Bool(data=True))
        self._use_satellite_pub.publish(Bool(data=False))
        self._use_image_pub.publish(Bool(data=False))
        self._use_lan_pub.publish(Bool(data=language))

    def _start_motion(self, checkpoint: IndoorCheckpoint):
        self._publish_modality(checkpoint)
        self._enable_pub.publish(Bool(data=True))
        with self._lock:
            self._motion_allowed = True
            self._stop_requested = False

    def _request_stop(self):
        with self._lock:
            self._motion_allowed = False
            self._stop_requested = True
        self._enable_pub.publish(Bool(data=False))
        self._publish_zero()

    def _resume_motion(self, checkpoint: IndoorCheckpoint):
        with self._lock:
            self._motion_allowed = True
            self._stop_requested = False
        self._publish_modality(checkpoint)
        self._enable_pub.publish(Bool(data=True))

    # ------------------------------------------------------------ SDK

    def _fetch_latest_scanned(self) -> Optional[int]:
        response = requests.post(self._param('checkpoint_list_url'), json={},
                                 timeout=self._param('http_timeout_s'))
        response.raise_for_status()
        payload = response.json()
        sdk_sequences = sorted(int(item['sequence']) for item in payload.get('checkpoints_list', []))
        ours = [c.sequence for c in self.track.checkpoints]
        if sdk_sequences != ours:
            # The track file is the route; the SDK's list is only compared.
            self.get_logger().warn(f'SDK checkpoint sequences {sdk_sequences} differ from '
                                   f'{self._param("route_file")} {ours}.')
        return int(payload.get('latest_scanned_checkpoint', 0))

    def _post_checkpoint_reached(self):
        response = requests.post(self._param('checkpoint_reached_url'), json={},
                                 timeout=self._param('http_timeout_s'))
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            payload = {}
        return response.status_code, payload

    def _verify(self, goal_handle, checkpoint: IndoorCheckpoint):
        """(reached, mission_completed, payload), as checkpoint_controller_node's loop."""
        if not self._param('confirm_with_sdk'):
            last = checkpoint.sequence == self.track.checkpoints[-1].sequence
            return True, last, {'message': 'accepted locally (confirm_with_sdk false)'}
        while not goal_handle.is_cancel_requested:
            status, payload = self._post_checkpoint_reached()
            if status == 200:
                return True, bool(payload.get('mission_completed', False)), payload
            if status == 400:
                self.get_logger().warn(f'SDK rejected {checkpoint.name}: {payload}')
                return False, False, payload
            if status == 503:
                self.get_logger().warn('Final stop not confirmed; holding zero and retrying.')
                self._request_stop()
                self._feedback(goal_handle, checkpoint.sequence, 0.0, 'confirming_stop')
                time.sleep(float(self._param('verification_retry_s')))
                continue
            raise RuntimeError(f'checkpoint-reached returned HTTP {status}')
        return False, False, {}

    # ------------------------------------------------------------ mission

    def _feedback(self, goal_handle, sequence, distance, state):
        fb = StartMission.Feedback()
        fb.current_checkpoint_sequence = int(sequence)
        fb.distance_m = float(distance)
        fb.state = state
        goal_handle.publish_feedback(fb)

    def _publish_route(self, pts):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = TRACK_FRAME
        for x, y in pts:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self._route_pub.publish(path)

    def _publish_carrot(self, x, y, bearing_deg):
        lat, lon = track_to_latlon(x, y)
        goal = NavSatFix()
        goal.latitude, goal.longitude = lat, lon
        self._goal_gps_pub.publish(goal)
        # Route direction at the carrot, compass convention: see
        # checkpoint_controller_node._publish_carrot for why not a constant.
        self._goal_compass_pub.publish(Float32(data=float(bearing_deg)))
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = TRACK_FRAME
        pt.point.x, pt.point.y = float(x), float(y)
        self._carrot_pub.publish(pt)

    def _follow(self, goal_handle, checkpoint: IndoorCheckpoint, radius_m: float) -> bool:
        """Carrot along the leg until within radius_m of the checkpoint (True) or canceled."""
        with self._lock:
            here = self._track_pose
        # point_at/project_forward work on (east, north) = track (x, y).
        pts = np.array(self.track.leg_route(checkpoint, here.x, here.y), dtype=float)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        self._publish_route(pts)
        lookahead = float(self._param('carrot_distance_m'))
        period = 1.0 / float(self._param('carrot_rate_hz'))
        s_proj, started, last_publish = 0.0, False, 0.0
        with self._odom_condition:
            while not goal_handle.is_cancel_requested:
                pose = self._track_pose
                remaining = math.hypot(checkpoint.x - pose.x, checkpoint.y - pose.y)
                self._current_distance_m = remaining
                if remaining <= radius_m:
                    return True
                s_proj = project_forward(pts, cum, pose.x, pose.y, s_proj)
                now = time.monotonic()
                if now - last_publish >= period:
                    last_publish = now
                    self._publish_carrot(*point_at(pts, cum, s_proj + lookahead))
                    if not started:
                        self._start_motion(checkpoint)
                        started = True
                    self._feedback(goal_handle, checkpoint.sequence, remaining, 'navigating')
                self._odom_condition.wait(timeout=period)
        return False

    def _execute_callback(self, goal_handle):
        result = StartMission.Result()
        try:
            resume = goal_handle.request.resume_from_latest_scanned
            latest = self._last_confirmed
            if self._param('confirm_with_sdk'):
                latest = self._fetch_latest_scanned()
            start_after = latest if resume else 0
            if not self._anchor_at(self.track.departure_pose(start_after)):
                raise RuntimeError(f'no odometry on {self._param("odom_topic")}: is ekf_local running?')

            with self._lock:
                self._mission_active = True
                self._motion_allowed = False
                self._stop_requested = True

            todo = [c for c in self.track.checkpoints if c.sequence > start_after]
            index = 0
            radius = float(self._param('checkpoint_proximity_m'))
            while index < len(todo):
                checkpoint = todo[index]
                self.get_logger().info(
                    f'Leg to {checkpoint.name} (seq {checkpoint.sequence}, {checkpoint.cone} cone) at '
                    f'({checkpoint.x:.1f}, {checkpoint.y:.1f}), via {list(checkpoint.via)}, '
                    f'arrival within {radius:.1f} m.')
                if not self._follow(goal_handle, checkpoint, radius):
                    goal_handle.canceled()
                    result.message = 'Mission canceled.'
                    return result
                self._request_stop()
                self._feedback(goal_handle, checkpoint.sequence, self._current_distance_m,
                               'confirming_checkpoint')
                reached, completed, payload = self._verify(goal_handle, checkpoint)
                if not reached:
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        result.message = 'Mission canceled.'
                        return result
                    tighter = max(float(self._param('min_checkpoint_proximity_m')), 0.5 * radius)
                    if tighter == radius:
                        # Already at the smallest radius around a point that is
                        # only dead-reckoned: re-posting from here cannot change
                        # the answer. Hold for the operator (cancel, drive to the
                        # cone, resume) instead of hammering the endpoint.
                        self._feedback(goal_handle, checkpoint.sequence, self._current_distance_m,
                                       'rejected_at_min_radius')
                        time.sleep(float(self._param('verification_retry_s')))
                    radius = tighter
                    self._resume_motion(checkpoint)
                    continue
                self._last_confirmed = checkpoint.sequence
                result.last_checkpoint_sequence = checkpoint.sequence
                self.get_logger().info(f'{checkpoint.name} confirmed: {payload.get("message", "")}')
                if completed:
                    break
                index += 1
                radius = float(self._param('checkpoint_proximity_m'))
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
    node = IndoorMissionNode()
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
