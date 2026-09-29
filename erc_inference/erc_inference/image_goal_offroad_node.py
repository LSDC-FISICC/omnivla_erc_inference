#!/usr/bin/env python3
"""Off-road image-goal missions: drive to where the goal photo was taken. No GPS, no map.

ERC 2026 off-road track (trails, slopes, gravel, rocks; the Verti-Arena kind of arena): each
mission is a target image. Which image is only known on the day, so it is a parameter,
read again at every start_mission:

    ros2 run erc_inference image_goal_offroad_node --ros-args -p goal_image:=/path/goal.jpg
    ros2 action send_goal --feedback /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

    # next mission, same node:
    ros2 param set /image_goal_offroad_node goal_image /path/other.jpg

goal_image: one image, several separated by commas (visited in that order), or a directory
(its images in name order).

The mission is homing_mission.HomingMission (no ROS, the same code test/homing_sim.py runs):
scan for the goal's scene, then stop-and-go hops along the bearing goal_homing.GoalMatcher
measures, until the parallax says the rover stands where the photo was taken; exploration
when the scene is not in view. This node supplies it with:

  pose       the local EKF (wheels + gyro), anchored at (0, 0, 0) when the goal is sent
             (indoor_mission_node's pseudo-fix, so carrot_controller_node can drive it).
  homing     SIFT matches of /erc/front_camera frames against the goal image, in a worker
             thread at homing_rate_hz, each with the pose at the frame's own stamp.
  motion     after each hop, a frame of the new stop against one of the last stop: the
             wheels can spin on gravel or against a rock while odometry counts metres.
  obstacles  /erc/free_space (free_space_node) for the local planner.
  tilt       /erc/imu_attitude for the safety envelope: stop, back off, mark it, go round.

Output as image_checkpoint_controller_node: the carrot out as /goal_gps + /goal_compass, or a
direct command (scans, pulses, back-offs) that overrides it; the safety envelope and the
acceleration limits; /cmd_vel from its own thread. This node is the only writer of /cmd_vel:
killing it takes the rover back.

Arrival: /checkpoint-reached with confirm_with_sdk (default true). A 400 is a rejection: the
mission closes in (a tighter parallax), up to max_rejections, then it stops there. No answer
at all (the endpoint missing for these missions): accepted locally, with a warning.
"""
import collections
import json
import math
import os
import threading
import time
from typing import List, Optional

import numpy as np

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import Bool, String

from erc_inference_msgs.action import StartMission

from erc_inference.goal_homing import DEFAULTS as MATCH_DEFAULTS
from erc_inference.goal_homing import GoalMatcher
from erc_inference.homing_mission import HOMING_DEFAULTS, MEASURE, HomingGoal, HomingMission
from erc_inference.indoor_mission_node import IndoorMissionNode, yaw_from_quaternion
from erc_inference.indoor_track import Pose2D, anchor
from erc_inference.safety_envelope import DEFAULTS as SAFETY_DEFAULTS
from erc_inference.safety_envelope import SafetyEnvelope, parameter_errors as safety_errors

# Off-road: the speed floor, straight reverse allowed (back-offs, a goal just behind), tilt stop
# ON (it is what stands between a boulder and a rolled rover), no obstacle stop (the planner
# goes round what free space maps; a stop in clutter deadlocked indoors, loop_patrol_sim). A
# missing attitude topic does not freeze the rover.
SAFETY_OFFROAD = {'safety.max_linear_vel': 0.25, 'safety.max_reverse_vel': 0.25, 'safety.obstacle_enabled': False,
                  'safety.tilt_enabled': True, 'safety.stop_if_attitude_stale': False}
LOCAL_OFFROAD = {'inflate_m': 0.3, 'unseen_extend_m': 0.1}

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def goal_paths(spec: str) -> List[str]:
    spec = (spec or '').strip()
    if not spec:
        return []
    if os.path.isdir(os.path.expanduser(spec)):
        d = os.path.expanduser(spec)
        return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(IMAGE_EXT)]
    return [os.path.expanduser(p.strip()) for p in spec.split(',') if p.strip()]


def image_to_rgb(msg: Image) -> np.ndarray:
    buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding == 'bgr8':
        return buf[:, :, ::-1].copy()
    return buf[:, :, :3].copy()


class ImageGoalOffroadNode(IndoorMissionNode):

    def __init__(self, node_name='image_goal_offroad_node', mission_timeout_s=900.0):
        # the arena file only gives the start pose, (0, 0, 0): the odometry frame where the goal is sent.
        # No SDK confirmation by default: the organisers judge by eye on the video (29-sept).
        super().__init__(node_name, route_file='offroad_arena.yaml', confirm_with_sdk=False)
        self.declare_parameter('goal_image', '')
        self.declare_parameter('image_topic', '/erc/front_camera')
        self.declare_parameter('free_space_topic', '/erc/free_space')
        self.declare_parameter('attitude_topic', '/erc/imu_attitude')
        self.declare_parameter('homing_topic', '/erc/homing')
        self.declare_parameter('mission_timeout_s', float(mission_timeout_s))
        self.declare_parameter('tick_rate_hz', 3.0)
        self.declare_parameter('homing_rate_hz', 3.0)
        # the picture is older than its stamp (TAREA1 6.2: 0.3-0.6 s); stop-and-go makes it
        # matter little -- decisions are taken standing still
        self.declare_parameter('image_content_lag_s', 0.45)
        self.declare_parameter('max_rejections', 3)
        for name, default in HOMING_DEFAULTS.items():
            self.declare_parameter(f'homing.{name}', default)
        for name, default in MATCH_DEFAULTS.items():
            self.declare_parameter(f'match.{name}', default)
        for name, default in SAFETY_DEFAULTS.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, SAFETY_OFFROAD.get(name, default))
        for name, default in LOCAL_OFFROAD.items():
            self.declare_parameter(f'local.{name}', default)
        errors = safety_errors(self._safety())
        if errors:
            raise ValueError('; '.join(errors))

        self.envelope = SafetyEnvelope()
        self.mission: Optional[HomingMission] = None
        self.matchers: List[GoalMatcher] = []
        self._override: Optional[tuple] = None
        self._yaws = []
        self._pose_hist = collections.deque()
        self._mission_lock = threading.RLock()
        self._frame = None                 # (ros stamp s, rgb)
        self._frame_seq = 0
        self._stop_frame = None            # a frame of the last stop, for the motion check
        self._last_note, self._last_note_t = '', -1e9
        self._match_ms = collections.deque(maxlen=50)
        self._preview_t, self._preview_log_t = 0.0, 0.0
        self._preview_key, self._preview_matcher = None, None
        self._t0 = time.monotonic()

        self.destroy_timer(self._output_timer)
        self._output_stop = threading.Event()
        threading.Thread(target=self._output_loop, daemon=True, name='cmd_vel').start()
        self._sensor_group = MutuallyExclusiveCallbackGroup()
        self._image_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, self._param('image_topic'), self._image_callback, 2,
                                 callback_group=self._image_group)
        self.create_subscription(LaserScan, self._param('free_space_topic'), self._scan_callback, 10,
                                 callback_group=self._sensor_group)
        self.create_subscription(Imu, self._param('attitude_topic'), self._attitude_callback, 20,
                                 callback_group=self._sensor_group)
        self._homing_pub = self.create_publisher(String, self._param('homing_topic'), 10)
        threading.Thread(target=self._homing_loop, daemon=True, name='homing').start()
        paths = goal_paths(self._param('goal_image'))
        self.get_logger().info(
            f'Off-road image-goal node ready; goal_image = {paths or "(not set yet)"}. Set it with '
            f'-p goal_image:=/path.jpg or `ros2 param set {self.get_fully_qualified_name()} goal_image ...`, '
            'then send start_mission.')

    def _now(self):
        return time.monotonic() - self._t0

    def _ros_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _safety(self):
        return {n: self.get_parameter(n).value for n in SAFETY_DEFAULTS}

    def _group(self, prefix, names):
        return {n: self.get_parameter(f'{prefix}.{n}').value for n in names}

    # -- goals ----------------------------------------------------------------

    def _load_goals(self):
        import cv2
        paths = goal_paths(self._param('goal_image'))
        if not paths:
            raise ValueError('goal_image is not set: -p goal_image:=/path/to/goal.jpg (or a comma list, or a directory)')
        goals, matchers = [], []
        for i, path in enumerate(paths):
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f'goal image not found or unreadable: {path}')
            m = GoalMatcher(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), **self._group('match', MATCH_DEFAULTS))
            n = 0 if m.goal_des is None else len(m.goal_des)
            self.get_logger().info(f'goal {i + 1}: {path} ({img.shape[1]}x{img.shape[0]}, {n} SIFT features)')
            if m.warning:
                self.get_logger().warn(f'goal {i + 1}: {m.warning}')
            if n < 200:
                self.get_logger().warn(f'goal {i + 1}: only {n} features -- a blurred or textureless image '
                                       'will match poorly')
            goals.append(HomingGoal(f'G{i + 1}', path, i + 1))
            matchers.append(m)
        return goals, matchers

    # -- pose ---------------------------------------------------------------

    def _odom_callback(self, msg: Odometry):
        p = msg.pose.pose
        odom = Pose2D(p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        with self._odom_condition:
            self._odom_pose = odom
            if self._track_from_odom is None:
                self._track_from_odom = anchor(self.track.start, odom)
            self._track_pose = self._track_from_odom.compose(odom)
            pose = self._track_pose
            t = self._now()
            self._yaws = [(tt, a) for tt, a in self._yaws if t - tt <= 1.0] + [(t, pose.yaw)]
            ts = self._ros_s()
            self._pose_hist.append((ts, pose))
            while self._pose_hist and ts - self._pose_hist[0][0] > 5.0:
                self._pose_hist.popleft()
            self._odom_condition.notify_all()
        self._publish_pose(pose, msg.header.stamp)

    def _pose_at(self, stamp_s):
        with self._lock:
            hist = list(self._pose_hist)
            latest = self._track_pose
        if not hist or stamp_s <= 0.0:
            return latest
        i = min(range(len(hist)), key=lambda k: abs(hist[k][0] - stamp_s))
        return hist[i][1] if abs(hist[i][0] - stamp_s) < 0.5 else latest

    def _turn_rate(self):
        h = self._yaws
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.3:
            return 0.0
        d = (h[-1][1] - h[0][1] + math.pi) % (2 * math.pi) - math.pi
        return abs(d) / (h[-1][0] - h[0][0])

    # -- perception -----------------------------------------------------------

    def _image_callback(self, msg: Image):
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        try:
            rgb = image_to_rgb(msg)
        except ValueError:
            return
        with self._lock:
            self._frame = (stamp if stamp > 0 else self._ros_s(), rgb)
            self._frame_seq += 1

    def _homing_loop(self):
        seen = 0
        while not self._output_stop.is_set() and rclpy.ok():
            period = 1.0 / float(self._param('homing_rate_hz'))
            t_start = time.monotonic()
            try:
                with self._lock:
                    frame, seq = self._frame, self._frame_seq
                with self._mission_lock:
                    m, matcher = self.mission, self._current_matcher()
                if frame is not None and seq != seen and m is not None and matcher is not None:
                    seen = seq
                    self._homing_step(m, matcher, *frame)
                elif frame is not None and matcher is None and time.monotonic() - self._preview_t >= 1.0:
                    self._preview_t = time.monotonic()
                    self._preview(*frame)
            except Exception as exc:
                if not rclpy.ok():
                    break
                self.get_logger().error(f'homing loop: {exc}', throttle_duration_sec=5.0)
            time.sleep(max(0.0, period - (time.monotonic() - t_start)))

    def _preview(self, stamp, rgb):
        """No mission running: match the first goal image anyway, 1 Hz, so the operator can see
        before sending the goal that the image loads and whether its scene is in view."""
        paths = goal_paths(self._param('goal_image'))
        if not paths:
            return
        if self._preview_key != paths[0]:
            import cv2
            img = cv2.imread(paths[0])
            self._preview_key = paths[0]
            self._preview_matcher = None if img is None else GoalMatcher(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB), **self._group('match', MATCH_DEFAULTS))
            if img is None:
                self.get_logger().warn(f'goal image not found or unreadable: {paths[0]}')
        if self._preview_matcher is None:
            return
        h = self._preview_matcher.match(rgb)
        self._homing_pub.publish(String(data=json.dumps({
            'stamp': stamp, 'goal': f'preview:{os.path.basename(paths[0])}', 'matches': h.matches,
            'inliers': h.inliers, 'parallax_px': round(h.parallax_px, 1) if h.ok else None,
            'bearing_deg': round(math.degrees(h.bearing), 1) if h.ok else None,
            'yaw_deg': round(math.degrees(h.yaw), 1) if h.ok else None, 'method': h.method})))
        if time.monotonic() - self._preview_log_t >= 15.0:
            self._preview_log_t = time.monotonic()
            seen = (f'in view: {h.inliers} inliers, parallax {h.parallax_px:.0f} px, bearing '
                    f'{math.degrees(h.bearing):+.0f} deg' if h.ok and h.inliers >= self._param('homing.min_inliers')
                    else f'not in view ({h.matches} raw matches)')
            self.get_logger().info(f'preview {os.path.basename(paths[0])}: {seen}')

    def _current_matcher(self):
        m = self.mission
        if m is None or m.done or m.index >= len(self.matchers):
            return None
        return self.matchers[m.index]

    def _homing_step(self, m: HomingMission, matcher: GoalMatcher, stamp, rgb):
        t0 = time.monotonic()
        h = matcher.match(rgb)
        self._match_ms.append(1000 * (time.monotonic() - t0))
        lag = float(self._param('image_content_lag_s'))
        pose = self._pose_at(stamp - lag)
        if pose is None:
            return
        t_cap = self._now() - (self._ros_s() - stamp) - lag       # capture time, mission clock
        motion = None
        with self._mission_lock:
            if self.mission is not m:
                return
            m.observe_homing(t_cap, pose.x, pose.y, pose.yaw, h)
            standing = m.state == MEASURE and m._stop_t0 is not None and t_cap >= m._stop_t0 + m.p['settle_s']
            want = standing and m.wants_motion_check and self._stop_frame is not None
            if standing and not m.wants_motion_check:
                self._stop_frame = rgb
        if want:
            prev = GoalMatcher(self._stop_frame, **self._group('match', MATCH_DEFAULTS))
            mh = prev.match(rgb)
            motion = mh.parallax_px if mh.ok else float('nan')
            with self._mission_lock:
                if self.mission is m:
                    m.observe_motion(t_cap, motion)
        self._homing_pub.publish(String(data=json.dumps({
            'stamp': stamp, 'goal': m.goal.name if m.goal else None, 'matches': h.matches, 'inliers': h.inliers,
            'parallax_px': round(h.parallax_px, 1) if h.ok else None,
            'bearing_deg': round(math.degrees(h.bearing), 1) if h.ok else None,
            'yaw_deg': round(math.degrees(h.yaw), 1) if h.ok else None,
            'method': h.method, 'motion_px': None if motion is None else round(motion, 1)})))

    def _scan_callback(self, msg: LaserScan):
        r = np.asarray(msg.ranges, float)
        b = np.degrees(msg.angle_min + np.arange(len(r)) * msg.angle_increment)
        hit = np.isfinite(r) & (r < msg.range_max - 0.05)
        r = np.where(np.isfinite(r), r, msg.range_max)
        t = self._now()
        self.envelope.observe_free_space(t, b, np.where(hit, r, msg.range_max), msg.range_max)
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        pose = self._pose_at(stamp)
        if pose is None:
            return
        with self._mission_lock:
            if self.mission is not None:
                self.mission.observe_scan(t, pose.x, pose.y, pose.yaw, b, r, hit)

    def _attitude_callback(self, msg: Imu):
        q = msg.orientation
        self.envelope.observe_attitude(self._now(), q.x, q.y, q.z, q.w)

    # -- output (as image_checkpoint_controller_node) ---------------------------

    def _output_loop(self):
        period = 1.0 / float(self._param('control_rate_hz'))
        nxt = time.monotonic()
        while not self._output_stop.is_set() and rclpy.ok():
            try:
                self._publish_output()
            except Exception as exc:
                if not rclpy.ok():
                    break
                self.get_logger().error(f'output loop: {exc}')
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
            if time.monotonic() - nxt > 1.0:
                nxt = time.monotonic()

    def _publish_output(self):
        now = self._now()
        with self._lock:
            active = self._mission_active and self._motion_allowed and not self._stop_requested
            if not active:
                v, w = 0.0, 0.0
            elif self._override is not None:
                v, w = self._override
            else:
                v, w = self._last_model_cmd.linear.x, self._last_model_cmd.angular.z
            hard, note = False, ''
            if active and (v != 0.0 or w != 0.0):
                v, w, hard, note = self.envelope.limit(now, v, w, self._safety())
            dt = 0.0 if self._last_output_time is None else now - self._last_output_time
            self._last_output_time = now
            if hard:
                self._shaper.reset()
            self._shaper.configure(*self._shaper_limits())
            linear, angular = self._shaper.step(v, w, dt)
        out = Twist()
        out.linear.x, out.angular.z = linear, angular
        self._cmd_pub.publish(out)
        if note and (note != self._last_note or now - self._last_note_t >= 2.0):
            self._last_note, self._last_note_t = note, now
            self.get_logger().info(note)

    # -- the mission ----------------------------------------------------------

    def _confirm_with_sdk(self, goal_handle):
        """-> 'accepted', 'rejected' or 'local' (no usable answer: accepted here, with a warning)."""
        if not self._param('confirm_with_sdk'):
            return 'local'
        for _ in range(5):
            if goal_handle.is_cancel_requested:
                return 'rejected'
            try:
                status, payload = self._post_checkpoint_reached()
            except Exception as exc:
                self.get_logger().warn(f'checkpoint-reached failed: {exc}')
                time.sleep(float(self._param('verification_retry_s')))
                continue
            if status == 200:
                return 'accepted'
            if status == 400:
                self.get_logger().warn(f'SDK rejected the arrival: {payload}')
                return 'rejected'
            self.get_logger().warn(f'checkpoint-reached returned HTTP {status}')
            time.sleep(float(self._param('verification_retry_s')))
        self.get_logger().warn('no usable answer from checkpoint-reached: arrival accepted locally')
        return 'local'

    def _execute_callback(self, goal_handle):
        result = StartMission.Result()
        t_start = self._now()
        try:
            goals, matchers = self._load_goals()
            with self._mission_lock:
                self.matchers = matchers
                self.mission = HomingMission(goals, (0.0, 0.0), self._group('local', LOCAL_OFFROAD), None,
                                             **self._group('homing', HOMING_DEFAULTS))
                self._stop_frame = None
            if not self._anchor_at(self.track.start):
                raise RuntimeError(f'no odometry on {self._param("odom_topic")}: is ekf_local running?')
            self.envelope.arm()
            with self._lock:
                self._mission_active = True
                self._motion_allowed = True
                self._stop_requested = False
                self._override = (0.0, 0.0)
            self._use_pose_pub.publish(Bool(data=True))
            self._enable_pub.publish(Bool(data=True))
            period = 1.0 / float(self._param('tick_rate_hz'))
            last_note, rejections, last_rate_log = 0, 0, t_start
            tilt_handled = False
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.message = 'Mission canceled.'
                    return result
                t = self._now()
                if t - t_start > float(self._param('mission_timeout_s')):
                    raise RuntimeError(f'mission timeout ({self._param("mission_timeout_s"):.0f} s)')
                if t - last_rate_log >= 30.0 and self._match_ms:
                    last_rate_log = t
                    self.get_logger().info(f'goal matching: {np.median(self._match_ms):.0f} ms per frame')
                with self._lock:
                    pose = self._track_pose
                with self._mission_lock:
                    m = self.mission
                    step = m.update(t, pose.x, pose.y, pose.yaw, self._turn_rate())
                    for n in m.notes[last_note:]:
                        self.get_logger().info(f'[homing] {n}')
                    last_note = len(m.notes)
                if m.done:
                    break
                g = m.goal
                if step.arrived:
                    with self._lock:
                        self._override = (0.0, 0.0)
                    self._publish_zero()
                    self._feedback(goal_handle, g.sequence, 0.0, f'confirming {g.name}')
                    time.sleep(1.5)
                    answer = self._confirm_with_sdk(goal_handle)
                    if answer == 'rejected':
                        rejections += 1
                        if rejections >= int(self._param('max_rejections')):
                            self.get_logger().warn(f'{g.name}: rejected {rejections} times; stopping here')
                            answer = 'local'
                    with self._mission_lock:
                        m.confirm(self._now(), answer != 'rejected')
                    if answer != 'rejected':
                        rejections = 0
                        result.last_checkpoint_sequence = g.sequence
                        self.get_logger().info(f'{g.name} reached ({answer}) in {self._now() - t_start:.0f} s '
                                               f'({m.index}/{len(m.goals)}).')
                    continue
                if self.envelope.tilted and not tilt_handled:
                    tilt_handled = True
                    with self._mission_lock:
                        m.block_ahead(self._now(), pose.x, pose.y, pose.yaw, f'tilt {self.envelope.tilt_deg:.0f} deg')
                    continue
                if not self.envelope.tilted:
                    tilt_handled = False
                with self._lock:
                    self._override = step.command
                if step.carrot is not None:
                    self._publish_carrot(*step.carrot)
                last = m._last
                par = f' parallax {last.h.parallax_px:.0f} px' if last is not None else ''
                self._feedback(goal_handle, g.sequence, -1.0, f'{step.state} {g.name}{par}')
                time.sleep(max(0.0, period - (self._now() - t)))
            goal_handle.succeed()
            result.success = True
            result.mission_completed = True
            result.message = f'All {len(self.mission.goals)} goal images reached in {self._now() - t_start:.0f} s.'
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
                self._override = None
            self._publish_zero()

    def destroy_node(self):
        self._output_stop.set()
        try:
            super().destroy_node()          # its last zero /cmd_vel fails once Ctrl-C shut rclpy down
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ImageGoalOffroadNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
