#!/usr/bin/env python3
"""Off-road checkpoint flags: reach, in order, the blue flags that are the checkpoints.

The organisers (29-sept): three blue flags serve as the checkpoints; within 1 m of each; the
goal photos come from any camera. The mission is flag_mission.FlagMission (no ROS, the same code
test/homing_sim.py's run_flags runs). This node supplies it with:

  pose     the local EKF, anchored at (0, 0, 0) when the goal is sent (the pseudo-fix, so
           carrot_controller_node drives it) -- as image_goal_offroad_node, whose output thread,
           safety envelope, free-space and attitude plumbing this node inherits.
  flags    /erc/flags (erc_perception flag_detector_node): bearing, range, apparent size.
  photos   optional (goal_image: one photo per checkpoint, in order, comma-separated or a
           directory): each frame's SIFT inliers against every photo (fundamental matrix, any
           camera), at photo_rate_hz; they rank which flag is which checkpoint.

    ros2 run erc_inference flag_checkpoint_node --ros-args -p goal_image:=cp1.jpg,cp2.jpg,cp3.jpg
    ros2 action send_goal --feedback /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

Without photos: -p checkpoints:=3 (the nearest unvisited flag first; the SDK's answer sorts them).
Resume after a restart: send resume_from_latest_scanned: true; the mission starts after the SDK's
latest scanned checkpoint.

Arrival: /checkpoint-reached. Accepted -> next. Rejected -> closer (once, while the range scale is
still assumed) or the next flag. No answer at all -> accepted locally, with a warning.
"""
import json
import time

import numpy as np

import rclpy
from std_msgs.msg import Bool, String

from erc_inference_msgs.action import StartMission

from erc_inference.flag_mission import FLAG_COMPETITION, FLAG_DEFAULTS, FlagMission
from erc_inference.goal_homing import DEFAULTS as MATCH_DEFAULTS
from erc_inference.goal_homing import GoalMatcher, photo_scores
from erc_inference.image_goal_offroad_node import LOCAL_OFFROAD, ImageGoalOffroadNode, goal_paths

# the image_goal_mission parameters worth exposing here (the rest keep flag_mission's defaults)
BASE_EXPOSED = {'arrive_m': 0.5, 'standoff_m': 0.55}


class FlagCheckpointNode(ImageGoalOffroadNode):

    def __init__(self):
        super().__init__(node_name='flag_checkpoint_node', mission_timeout_s=1800.0)   # the 30-minute window
        self.declare_parameter('flags_topic', '/erc/flags')
        self.declare_parameter('checkpoints', 3)
        self.declare_parameter('photo_rate_hz', 1.0)
        self.declare_parameter('max_rejections_per_checkpoint', 8)
        for name, default in FLAG_DEFAULTS.items():
            self.declare_parameter(f'flag.{name}', FLAG_COMPETITION.get(name, default))
        for name, default in BASE_EXPOSED.items():
            self.declare_parameter(f'flag.{name}', FLAG_COMPETITION.get(name, default))
        self._last_photo = 0.0
        self.create_subscription(String, self._param('flags_topic'), self._flags_callback, 10,
                                 callback_group=self._sensor_group)
        self.get_logger().info('Flag checkpoint node ready: flags on /erc/flags; goal photos '
                               f'{goal_paths(self._param("goal_image")) or "(none: nearest flag first)"}.')

    # -- goals: photos are optional hints -------------------------------------------

    def _load_goals(self):
        import cv2
        paths = goal_paths(self._param('goal_image'))
        matchers = []
        for i, path in enumerate(paths):
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f'goal photo not found or unreadable: {path}')
            m = GoalMatcher(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), **self._group('match', MATCH_DEFAULTS))
            n = 0 if m.goal_des is None else len(m.goal_des)
            self.get_logger().info(f'checkpoint {i + 1} photo: {path} ({img.shape[1]}x{img.shape[0]}, {n} SIFT features)')
            matchers.append(m)
        n_cp = len(paths) if paths else int(self._param('checkpoints'))
        return n_cp, matchers

    def _current_matcher(self):
        m = self.mission
        if m is None or m.done or not self.matchers:
            return None
        return self.matchers[0]          # a token: _homing_step scores every photo

    def _homing_step(self, m, matcher, stamp, rgb):
        now = time.monotonic()
        if now - self._last_photo < 1.0 / float(self._param('photo_rate_hz')):
            return
        self._last_photo = now
        scores = photo_scores(self.matchers, rgb)
        lag = float(self._param('image_content_lag_s'))
        pose = self._pose_at(stamp - lag)
        if pose is None:
            return
        t_cap = self._now() - (self._ros_s() - stamp) - lag
        with self._mission_lock:
            if self.mission is m:
                m.observe_scene(t_cap, pose.x, pose.y, pose.yaw, scores)
        self._homing_pub.publish(String(data=json.dumps({'stamp': stamp, 'photo_inliers': [s[0] for s in scores],
                                                         'photo_bearing_deg': [round(s[1], 1) for s in scores]})))

    def _preview(self, stamp, rgb):
        """No mission running: the photo scores of this view, so the operator sees them work."""
        if self.mission is not None and not self.mission.done:
            return
        paths = goal_paths(self._param('goal_image'))
        if not paths:
            return
        if self._preview_key != tuple(paths):
            import cv2
            self._preview_key = tuple(paths)
            self._preview_matcher = []
            for pth in paths:
                img = cv2.imread(pth)
                if img is None:
                    self.get_logger().warn(f'goal photo not found or unreadable: {pth}')
                    continue
                self._preview_matcher.append(GoalMatcher(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                                         **self._group('match', MATCH_DEFAULTS)))
        if not self._preview_matcher:
            return
        scores = photo_scores(self._preview_matcher, rgb)
        self._homing_pub.publish(String(data=json.dumps({'stamp': stamp, 'photo_inliers': [s[0] for s in scores],
                                                         'preview': True})))
        if time.monotonic() - self._preview_log_t >= 15.0:
            self._preview_log_t = time.monotonic()
            self.get_logger().info('preview, photo inliers in this view: ' + ', '.join(
                f'CP{i + 1} {s[0]}' for i, s in enumerate(scores)) + f' (a match is >= {self._param("flag.photo_min_inliers")})')

    # -- flags -----------------------------------------------------------------------------

    def _flags_callback(self, msg: String):
        try:
            doc = json.loads(msg.data)
            dets = doc.get('detections', [])
        except (ValueError, AttributeError):
            return
        stamp = float(doc.get('stamp', 0.0))
        lag = float(self._param('image_content_lag_s'))
        pose = self._pose_at(stamp - lag)
        if pose is None:
            return
        t_cap = self._now() - (self._ros_s() - stamp) - lag if stamp > 0 else self._now()
        with self._mission_lock:
            m = self.mission
            if m is None or m.done:
                return
            for d in dets:
                m.observe_cone(pose.x, pose.y, pose.yaw, 'blue', float(d['bearing_deg']), float(d['range_m']),
                               t=t_cap, ang_height=d.get('ang_height_rad'))

    # -- the mission -----------------------------------------------------------------------

    def _execute_callback(self, goal_handle):
        result = StartMission.Result()
        t_start = self._now()
        try:
            n_cp, matchers = self._load_goals()
            params = self._group('flag', list(FLAG_DEFAULTS) + list(BASE_EXPOSED))
            with self._mission_lock:
                self.matchers = matchers
                self.mission = FlagMission(n_cp, (0.0, 0.0), self._group('local', LOCAL_OFFROAD), None, **params)
            if not self._anchor_at(self.track.start):
                raise RuntimeError(f'no odometry on {self._param("odom_topic")}: is ekf_local running?')
            if goal_handle.request.resume_from_latest_scanned:
                # a restarted node: carry on after the SDK's latest scanned checkpoint (the flags
                # seen before are forgotten; the accepted ones are simply not asked for again)
                try:
                    latest = self._fetch_latest_scanned()
                except Exception as exc:
                    self.get_logger().warn(f'checkpoints-list failed ({exc})')
                    latest = 0
                if latest:
                    with self._mission_lock:
                        self.mission.index = min(int(latest), len(self.mission.goals))
                        self.mission._begin_leg(self._now())
                    self.get_logger().info(f'resuming after the SDK\'s latest scanned checkpoint: {latest}')
                else:
                    self.get_logger().warn('resume asked, but the SDK reports no scanned checkpoint: from CP1')
            self.envelope.arm()
            with self._lock:
                self._mission_active = True
                self._motion_allowed = True
                self._stop_requested = False
                self._override = (0.0, 0.0)
            self._use_pose_pub.publish(Bool(data=True))
            self._enable_pub.publish(Bool(data=True))
            period = 1.0 / float(self._param('tick_rate_hz'))
            last_note, rejections, tilt_handled = 0, 0, False
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
                with self._mission_lock:
                    m = self.mission
                    step = m.update(t, pose.x, pose.y, pose.yaw, self._turn_rate())
                    for n in m.notes[last_note:]:
                        self.get_logger().info(f'[flags] {n}')
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
                        if rejections >= int(self._param('max_rejections_per_checkpoint')):
                            self.get_logger().warn(f'{g.name}: rejected {rejections} times; moving on')
                            answer = 'local'
                    with self._mission_lock:
                        m.confirm(self._now(), answer != 'rejected')
                    if answer != 'rejected':
                        rejections = 0
                        result.last_checkpoint_sequence = g.sequence
                        self.get_logger().info(f'{g.name} reached ({answer}) at {self._now() - t_start:.0f} s '
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
                tgt = m.target
                dist = float(np.hypot(*(m.est(tgt) - (pose.x, pose.y)))) if tgt is not None else -1.0
                self._feedback(goal_handle, g.sequence, dist, f'{step.state} {g.name} ({len(m.flags)} flags seen)')
                time.sleep(max(0.0, period - (self._now() - t)))
            goal_handle.succeed()
            result.success = True
            result.mission_completed = True
            result.message = f'All {len(self.mission.goals)} flags reached in {self._now() - t_start:.0f} s.'
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


def main(args=None):
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    rclpy.init(args=args)
    node = FlagCheckpointNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
