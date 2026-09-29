#!/usr/bin/env python3
"""checkpoint_controller_node for image-goal missions: reach, in order, the cones the goal images show.

ERC 2026 NYU indoor: the mission is a list of images (config/indoor_nyu_goals.yaml, the
organisers' cone photos), no coordinates, no GPS; the corridor walls are known
(config/indoor_nyu_track.yaml), the cones move, chairs and open doors may appear; 30
minutes, fully autonomous, nothing tested on the rover beforehand.

The mission itself is loop_patrol.LoopPatrolMission (no ROS, the same code
test/loop_patrol_sim.py runs): patrol the known loop, map every cone seen (any colour),
visit them in the images' order the shorter way round, scanning at corners; local
replanning around what /erc/free_space maps, known walls as the static layer. This node
supplies it with:

  pose       the local EKF (wheels + gyro) anchored to the track at the start pose
             (indoor_mission_node), then corrected against the known walls
             (loop_patrol.WallLocalizer, point-to-wall matching on /erc/free_space). The
             corrected pose is what goes out as the pseudo-fix, so the carrot controller
             and the mission agree on where the rover is.
  cones      /erc/cones from erc_perception's cone_detector_node (JSON).
  goals      each image's cone colour, read from the image (erc_perception.cones
             .classify_goal_image). It refuses to start if an image is ambiguous.

and turns its output into commands: the carrot (the route point carrot_distance_m
ahead) out as /goal_gps + /goal_compass for the controller (carrot_controller_node, or
an OmniVLA node), or a direct command (scan turns, holds) that overrides it. Every
command then passes safety_envelope (speed caps, obstacle stop that keeps turning) and
the acceleration limits. Arrival -> /checkpoint-reached as outdoors (confirm_with_sdk);
a rejection makes the mission close in.

    ros2 run erc_inference image_checkpoint_controller_node
    ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

The rover must stand on the start pose of the track file when the goal is sent (NYU:
next to the orange cone, facing the south corridor's east end). This node is the only
writer of /cmd_vel: killing it takes the rover back.

Resume: the node saves its anchor, wall correction, cone map and next goal to state_file
every 2 s. After killing it (an intervention; the rover may be driven by hand meanwhile --
ekf_local keeps tracking it as long as the launch keeps running), restart it and send
resume_from_latest_scanned: true to carry on. The SDK's latest scanned checkpoint wins if it
is further on. Without a usable state (none, too old, odometry restarted) it anchors at the
start pose and says so.
"""
import collections
import json
import math
import os
import threading
import time
from typing import Optional

import numpy as np
import yaml

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, String

from erc_inference_msgs.action import StartMission

from erc_inference.image_goal_mission import Goal
from erc_inference.indoor_mission_node import IndoorMissionNode, yaw_from_quaternion
from erc_inference.indoor_track import Pose2D, anchor
from erc_inference.loop_patrol import DEFAULTS as PATROL_DEFAULTS
from erc_inference.loop_patrol import LOCAL_INDOOR, SAFETY_INDOOR, KnownMap, LoopPatrolMission, WallLocalizer
from erc_inference.safety_envelope import DEFAULTS as SAFETY_DEFAULTS
from erc_inference.safety_envelope import SafetyEnvelope, parameter_errors as safety_errors

# Indoor safety and planner settings: loop_patrol.SAFETY_INDOOR / LOCAL_INDOOR (shared with the simulator).


def load_goals(path, logger):
    import cv2
    from erc_perception.cones import classify_goal_image
    with open(path) as f:
        doc = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    goals = []
    for i, item in enumerate(doc['goals']):
        img_path = item['image'] if os.path.isabs(item['image']) else os.path.join(base, item['image'])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'goal image not found: {img_path}')
        cls, shares = classify_goal_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if cls is None:
            raise ValueError(f"goal {item['name']}: cannot tell the cone colour of {img_path} "
                             f"(class shares {shares})")
        goals.append(Goal(str(item['name']), cls, i + 1, is_start_cone=bool(item.get('start_cone', False))))
        logger.info(f"goal {i + 1} {item['name']}: {os.path.basename(img_path)} -> {cls} "
                    f"({100 * shares[cls]:.0f}% of the image)" + (' [start cone]' if goals[-1].is_start_cone else ''))
    return goals


class ImageCheckpointControllerNode(IndoorMissionNode):

    def __init__(self):
        super().__init__('image_checkpoint_controller_node')
        share = get_package_share_directory('erc_inference')
        self.declare_parameter('goals_file', os.path.join(share, 'config', 'indoor_nyu_goals.yaml'))
        self.declare_parameter('cones_topic', '/erc/cones')
        self.declare_parameter('free_space_topic', '/erc/free_space')
        self.declare_parameter('attitude_topic', '/erc/imu_attitude')
        self.declare_parameter('mission_timeout_s', 1800.0)        # the competition's 30 minutes
        # Where the mission's memory is saved every 2 s (anchor, wall correction, cone map, next
        # goal), so resume_from_latest_scanned: true can carry on after the node is restarted.
        self.declare_parameter('state_file', os.path.expanduser('~/.ros/erc_indoor_mission_state.json'))
        self.declare_parameter('state_max_age_s', 7200.0)
        self.declare_parameter('localize', True)
        self.declare_parameter('tick_rate_hz', 3.0)
        for name, default in PATROL_DEFAULTS.items():
            self.declare_parameter(f'patrol.{name}', default)
        for name, default in SAFETY_DEFAULTS.items():
            self.declare_parameter(name, SAFETY_INDOOR.get(name, default))
        for name, default in LOCAL_INDOOR.items():
            self.declare_parameter(f'local.{name}', default)
        errors = safety_errors(self._safety())
        if errors:
            raise ValueError('; '.join(errors))

        with open(self._param('route_file')) as f:
            self.known = KnownMap.from_yaml(yaml.safe_load(f))
        self.goals = load_goals(self._param('goals_file'), self.get_logger())
        self.localizer = WallLocalizer(self.known)
        self.envelope = SafetyEnvelope()
        self.mission: Optional[LoopPatrolMission] = None
        self._override: Optional[tuple] = None
        self._raw_pose: Optional[Pose2D] = None
        self._yaws = []
        # (ros time s, corrected pose) for the last few seconds: perception is matched against
        # the pose at its own stamp, not at whenever its callback got to run (the executor
        # delayed callbacks by up to ~1 s; mid-turn that is ~20 deg of error in the matching)
        self._pose_hist = collections.deque()
        self._scan_ages = []
        self._last_age_log = 0.0
        self._mission_lock = threading.RLock()
        self._last_save = -1e9
        self._last_note, self._last_note_t = '', -1e9

        self._t0 = time.monotonic()
        # /cmd_vel from a dedicated thread, not an executor timer: with the mission's long
        # action callback running, rclpy's executor left the 10 Hz timer unserved for up to
        # 1.5-2.6 s at a time (run_e2e_images.sh + a stack-dumping watchdog, 28-sept) --
        # worker threads idle, the executor loop busy. Publishing is thread-safe.
        self.destroy_timer(self._output_timer)
        self._output_stop = threading.Event()
        threading.Thread(target=self._output_loop, daemon=True, name='cmd_vel').start()
        # perception in its own group: it waits on the mission lock while the mission plans,
        # and must not hold up odometry or the /cmd_vel timer (see indoor_mission_node)
        self._sensor_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(String, self._param('cones_topic'), self._cones_callback, 10,
                                 callback_group=self._sensor_group)
        self.create_subscription(LaserScan, self._param('free_space_topic'), self._scan_callback, 10,
                                 callback_group=self._sensor_group)
        self.create_subscription(Imu, self._param('attitude_topic'), self._attitude_callback, 20,
                                 callback_group=self._sensor_group)
        self.get_logger().info(
            f'Image checkpoint controller ready: {len(self.goals)} goals, {len(self.known.corridors)} corridors, '
            f'loop {self.known.ring_len:.0f} m, {len(self.known.wall_a)} known walls. Waiting for start_mission.')

    def _now(self):
        """One time base for the mission, the planner, the envelope and the logs (s since start)."""
        return time.monotonic() - self._t0

    # -- parameters ---------------------------------------------------------

    def _safety(self):
        return {n: self.get_parameter(n).value for n in SAFETY_DEFAULTS}

    def _patrol(self):
        return {n: self.get_parameter(f'patrol.{n}').value for n in PATROL_DEFAULTS}

    # -- pose: anchor + wall correction -------------------------------------

    def _odom_callback(self, msg: Odometry):
        p = msg.pose.pose
        odom = Pose2D(p.position.x, p.position.y, yaw_from_quaternion(p.orientation))
        with self._odom_condition:
            self._odom_pose = odom
            if self._track_from_odom is None:
                self._track_from_odom = anchor(self.track.start, odom)
            raw = self._track_from_odom.compose(odom)
            self._raw_pose = raw
            x, y, yaw = self.localizer.apply(raw.x, raw.y, raw.yaw)
            self._track_pose = Pose2D(x, y, yaw)
            pose = self._track_pose
            t = self._now()
            self._yaws = [(tt, a) for tt, a in self._yaws if t - tt <= 1.0] + [(t, yaw)]
            ts = self._ros_s()
            self._pose_hist.append((ts, pose))
            while self._pose_hist and ts - self._pose_hist[0][0] > 5.0:
                self._pose_hist.popleft()
            self._odom_condition.notify_all()
        self._publish_pose(pose, msg.header.stamp)

    def _ros_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _pose_at(self, stamp_s):
        """The corrected pose at a ROS stamp (nearest sample within the last 5 s), else the latest."""
        with self._lock:
            hist = list(self._pose_hist)
            latest = self._track_pose
        if not hist or stamp_s <= 0.0:
            return latest
        i = min(range(len(hist)), key=lambda k: abs(hist[k][0] - stamp_s))
        return hist[i][1] if abs(hist[i][0] - stamp_s) < 0.5 else latest

    def _turn_rate_dps(self):
        h = self._yaws
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.3:
            return 0.0
        d = (h[-1][1] - h[0][1] + math.pi) % (2 * math.pi) - math.pi
        return abs(math.degrees(d)) / (h[-1][0] - h[0][0])

    def _anchor_at(self, track_pose: Pose2D) -> bool:
        ok = super()._anchor_at(track_pose)
        if ok:
            self.localizer = WallLocalizer(self.known)     # a fresh anchor has no correction yet
        return ok

    # -- perception ---------------------------------------------------------

    def _scan_callback(self, msg: LaserScan):
        r = np.asarray(msg.ranges, float)
        b = np.degrees(msg.angle_min + np.arange(len(r)) * msg.angle_increment)
        hit = np.isfinite(r) & (r < msg.range_max - 0.05)
        r = np.where(np.isfinite(r), r, msg.range_max)
        t = self._now()
        self.envelope.observe_free_space(t, b, np.where(hit, r, msg.range_max), msg.range_max)
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        age = self._ros_s() - stamp if stamp > 0 else float('nan')
        self._scan_ages.append(age)
        if t - self._last_age_log >= 30.0 and self._scan_ages:
            a = np.array([v for v in self._scan_ages if np.isfinite(v)])
            if len(a):
                self.get_logger().info(f'free space age on arrival: p50 {np.median(a):.2f} s, max {a.max():.2f} s '
                                       f'({len(a)} profiles)')
            self._scan_ages, self._last_age_log = [], t
        pose = self._pose_at(stamp)
        if pose is None:
            return
        with self._mission_lock:
            if self.mission is not None:
                self.mission.observe_scan(t, pose.x, pose.y, pose.yaw, b, r, hit)
            if self._param('localize'):
                self.localizer.observe(pose.x, pose.y, pose.yaw, b, r, hit, self._turn_rate_dps())

    def _cones_callback(self, msg: String):
        try:
            doc = json.loads(msg.data)
            dets = doc.get('detections', [])
        except (ValueError, AttributeError):
            return
        pose = self._pose_at(float(doc.get('stamp', 0.0)))
        if pose is None:
            return
        t = self._now()
        with self._mission_lock:
            if self.mission is None:
                return
            for d in dets:
                self.mission.observe_cone(pose.x, pose.y, pose.yaw, d['color'], float(d['bearing_deg']),
                                          float(d['range_m']), t)

    def _attitude_callback(self, msg: Imu):
        q = msg.orientation
        self.envelope.observe_attitude(self._now(), q.x, q.y, q.z, q.w)

    # -- output: override, then the safety envelope, then the shaper -------

    def _output_loop(self):
        period = 1.0 / float(self._param('control_rate_hz'))
        nxt = time.monotonic()
        while not self._output_stop.is_set() and rclpy.ok():
            try:
                self._publish_output()
            except Exception as exc:   # a context gone at shutdown; anything else must not kill /cmd_vel
                if not rclpy.ok():
                    break
                self.get_logger().error(f'output loop: {exc}')
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
            if time.monotonic() - nxt > 1.0:        # fell far behind: do not burst to catch up
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
        # Outside the lock, and throttled by hand: rclpy's logger inspects the caller's frame
        # (source files, realpath) on EVERY call, throttled or not. Called here at 10 Hz under
        # the lock it starved /cmd_vel to 2.8 Hz with 3.4 s gaps (run_e2e_images.sh, 28-sept).
        if note and (note != self._last_note or now - self._last_note_t >= 2.0):
            self._last_note, self._last_note_t = note, now
            self.get_logger().info(note)

    # -- the mission --------------------------------------------------------

    def _publish_pose_modality(self):
        self._use_pose_pub.publish(Bool(data=True))
        self._use_satellite_pub.publish(Bool(data=False))
        self._use_image_pub.publish(Bool(data=False))
        self._use_lan_pub.publish(Bool(data=False))

    # -- resume ---------------------------------------------------------------

    def _sdk_latest(self):
        if not self._param('confirm_with_sdk'):
            return None
        try:
            return self._fetch_latest_scanned()
        except Exception as exc:
            self.get_logger().warn(f'checkpoints-list failed ({exc}); resuming from the saved state alone')
            return None

    def _save_state(self):
        self._last_save = self._now()
        with self._lock:
            anchor_ = self._track_from_odom
            odom = self._odom_pose
        with self._mission_lock:
            if self.mission is None or anchor_ is None or odom is None:
                return
            doc = {'saved_at': time.time(), 'mission': self.mission.export_state(),
                   'anchor': [anchor_.x, anchor_.y, anchor_.yaw],
                   'localizer': [self.localizer.dx, self.localizer.dy, self.localizer.dyaw],
                   'odom': [odom.x, odom.y, odom.yaw]}
        path = self._param('state_file')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(doc, f)
        os.replace(tmp, path)                   # never a half-written file

    def _resume(self) -> bool:
        """Carry on from the saved state. False (and why, logged) when there is none to trust."""
        path = self._param('state_file')
        try:
            with open(path) as f:
                doc = json.load(f)
        except (OSError, ValueError) as exc:
            self.get_logger().warn(f'no saved mission state ({path}: {exc})')
            return False
        age = time.time() - float(doc.get('saved_at', 0.0))
        if age > float(self._param('state_max_age_s')):
            self.get_logger().warn(f'saved mission state is {age / 60:.0f} min old: not used')
            return False
        with self._lock:
            odom = self._odom_pose
        if odom is None:
            return False
        so = doc['odom']
        # ekf_local restarted (terminal 2 relaunched): its frame starts again at the origin, and the
        # saved anchor would put the rover somewhere it is not
        if math.hypot(so[0], so[1]) > 1.0 and math.hypot(odom.x, odom.y) < 0.3:
            self.get_logger().warn('the odometry was restarted since the state was saved: not used')
            return False
        with self._lock:
            self._track_from_odom = Pose2D(*doc['anchor'])
        self.localizer = WallLocalizer(self.known)
        self.localizer.dx, self.localizer.dy, self.localizer.dyaw = doc['localizer']
        latest = self._sdk_latest()
        first = latest if latest else None
        with self._mission_lock:
            self.mission.import_state(doc['mission'], first)
            idx = self.mission.index
        self.get_logger().info(f'Resumed from {path} ({age:.0f} s old): goal {idx + 1}/{len(self.goals)}'
                               + (f', SDK latest scanned {latest}' if latest is not None else '')
                               + f', {len(doc["mission"]["cones"])} cones mapped.')
        return True

    def _confirm_with_sdk(self, goal_handle):
        """-> (accepted, mission_completed). Local acceptance when confirm_with_sdk is false."""
        if not self._param('confirm_with_sdk'):
            return True, False
        for _ in range(30):
            if goal_handle.is_cancel_requested:
                return False, False
            try:
                status, payload = self._post_checkpoint_reached()
            except Exception as exc:
                self.get_logger().warn(f'checkpoint-reached failed: {exc}')
                time.sleep(float(self._param('verification_retry_s')))
                continue
            if status == 200:
                return True, bool(payload.get('mission_completed', False))
            if status == 400:
                self.get_logger().warn(f'SDK rejected the arrival: {payload}')
                return False, False
            if status == 503:
                time.sleep(float(self._param('verification_retry_s')))
                continue
            self.get_logger().warn(f'checkpoint-reached returned HTTP {status}')
            time.sleep(float(self._param('verification_retry_s')))
        return False, False

    def _execute_callback(self, goal_handle):
        result = StartMission.Result()
        t_start = self._now()
        try:
            lp_params = {n: self.get_parameter(f'local.{n}').value for n in LOCAL_INDOOR}
            with self._mission_lock:
                self.mission = LoopPatrolMission(self.goals, self.known, (self.track.start.x, self.track.start.y),
                                                 'red_orange', lp_params, None, **self._patrol())
            if not (goal_handle.request.resume_from_latest_scanned and self._resume()):
                if not self._anchor_at(self.track.start):
                    raise RuntimeError(f'no odometry on {self._param("odom_topic")}: is ekf_local running?')
                if goal_handle.request.resume_from_latest_scanned:
                    latest = self._sdk_latest()
                    if latest:
                        with self._mission_lock:
                            self.mission.index = min(latest, len(self.goals))
                    self.get_logger().warn(
                        f'Resume WITHOUT saved state: anchored at the start pose, continuing at goal '
                        f'{self.mission.index + 1}. The rover must be at the start, facing the corridor.')
            self.envelope.arm()
            with self._lock:
                self._mission_active = True
                self._motion_allowed = True
                self._stop_requested = False
                self._override = (0.0, 0.0)
            self._publish_pose_modality()
            self._enable_pub.publish(Bool(data=True))
            period = 1.0 / float(self._param('tick_rate_hz'))
            last_note = 0
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.message = 'Mission canceled.'
                    return result
                t = self._now()
                if t - t_start > float(self._param('mission_timeout_s')):
                    raise RuntimeError(f'mission timeout ({self._param("mission_timeout_s"):.0f} s)')
                with self._lock:
                    pose = self._track_pose
                if t - self._last_save >= 2.0:
                    self._save_state()
                with self._mission_lock:
                    m = self.mission
                    step = m.update(t, pose.x, pose.y, pose.yaw, math.radians(self._turn_rate_dps()))
                    for n in m.notes[last_note:]:
                        self.get_logger().info(f'[patrol] {n}')
                    last_note = len(m.notes)
                if m.done:
                    break
                g = m.goal
                if step.arrived:
                    with self._lock:
                        self._override = (0.0, 0.0)
                    self._publish_zero()
                    self._feedback(goal_handle, g.sequence, 0.0, f'confirming {g.name}')
                    time.sleep(1.5)                   # stand still before asking
                    accepted, completed = self._confirm_with_sdk(goal_handle)
                    with self._mission_lock:
                        m.confirm(self._now(), accepted)
                    if accepted:
                        result.last_checkpoint_sequence = g.sequence
                        self.get_logger().info(f'{g.name} confirmed ({m.index}/{len(self.goals)}).')
                    if completed:
                        break
                    continue
                if self.envelope.tilted and not getattr(self, '_tilt_handled', False):
                    self._tilt_handled = True
                    with self._mission_lock:
                        m.block_ahead(self._now(), pose.x, pose.y, pose.yaw, f'tilt {self.envelope.tilt_deg:.0f} deg')
                    continue
                if not self.envelope.tilted:
                    self._tilt_handled = False
                with self._lock:
                    self._override = step.command
                if step.carrot is not None:
                    self._publish_carrot(*step.carrot)
                target = m.target.estimate() if m.target is not None else None
                dist = math.hypot(target[0] - pose.x, target[1] - pose.y) if target is not None else -1.0
                self._feedback(goal_handle, g.sequence, dist, f'{step.state} {g.name}')
                time.sleep(max(0.0, period - (self._now() - t)))
            goal_handle.succeed()
            result.success = True
            result.mission_completed = True
            result.message = f'All {len(self.goals)} goals reached in {self._now() - t_start:.0f} s.'
            return result
        except Exception as exc:
            self.get_logger().error(f'Mission failed: {exc}')
            goal_handle.abort()
            result.success = False
            result.message = str(exc)
            return result
        finally:
            try:
                self._save_state()
            except Exception as exc:  # never let saving mask the mission's own outcome
                self.get_logger().warn(f'could not save the mission state: {exc}')
            self._enable_pub.publish(Bool(data=False))
            with self._lock:
                self._mission_active = False
                self._motion_allowed = False
                self._stop_requested = True
                self._override = None
            self._publish_zero()


    def destroy_node(self):
        self._output_stop.set()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageCheckpointControllerNode()
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
