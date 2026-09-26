#!/usr/bin/env python3
"""Glue between checkpoint_controller_node and Nav2's MPPI (launch/mission_nav2.launch.py).

The architecture stays the same as mission_carrot: A* on OSM is the global plan,
checkpoint_controller_node owns the mission, arrival and the only /cmd_vel. What
changes is the local layer: instead of carrot + polar control (+ side-step), Nav2's
controller_server runs MPPI on a local costmap built from the DA3 free-space profile.

This node does four things:
  1. ROUTE -> FollowPath. The leg's route (/erc/global_route, or /erc/local_route
     when checkpoint_controller_node also runs local_replan), densified to 0.1 m,
     goes to controller_server as a FollowPath goal -- while /enable_inference is
     true. Disabled (arrival, cancel, operator) -> the goal is cancelled.
  2. NEW LEG -> clear the costmap. leg_local restarts at every leg; obstacles
     marked in the previous leg's frame would land in the wrong place.
  3. PROFILE -> costmap. /erc/free_space_leg (the DA3 profile, already re-framed
     into rover_leg by checkpoint_controller_node) cropped to +-crop_deg and
     republished for the costmap's obstacle layer.
  4. MPPI's COMMAND -> the rover. MPPI knows nothing of this rover's floor: below
     ~0.25 m/s commanded it does not move (mission_16sept), and in place it needs
     ~0.15 rad/s to break away. snap_command() puts MPPI's output onto what the
     rover can do and publishes it on /omnivla/cmd_vel, which
     checkpoint_controller_node gates and ramps exactly as it does the carrot's.

  5. PREDICTED POSE. MPPI decides for the state it is given, and its command
     acts ~1.3 s later on this rover; it has no delay model. So this node
     publishes TF leg_local -> rover_pred: the pose now (leg_local -> rover_leg)
     rolled forward through the commands sent in the last predict_delay_s, with
     the measured plant gains. The costmap's robot_base_frame is rover_pred, so
     MPPI plans from where the rover will be when its command lands (a Smith
     predictor). Without it MPPI weaved +-0.7 m with a ~25 s period (e2e, hedge)
     -- the same weave mission_wuhan_hard showed in the field.
     The ~1.3 s is the total command -> observed-motion delay; whichever reading
     of it is true (stale telemetry + short actuation, or fresh telemetry +
     long actuation; Mission16sept 7.2), the horizon to roll forward is the same.

  6. YAW GAIN LEARNED ONLINE. The rover executes k_w x the commanded yaw rate, and k_w
     is per unit: 0.36 moving on mission_16sept, 1.23 on mission_24sept_Nav2_circles.
     With the wrong value the compensation and the predictor are both off by 3x; that
     is what made MPPI circle there (rover_pred predicted 8 deg where the rover turned
     26). With k_w_auto the node measures it while driving: yaw rate observed on
     yaw_rate_topic / the command sent predict_delay_s earlier, median over the last
     k_w_window_s, separately moving and in place.
  7. TURN IN PLACE FIRST. Below v_floor the rover does not move, so every large
     correction MPPI asks becomes a circle of ~1.4 m at 0.25 m/s, overshot by the
     delay. When the route is more than turn_enter_deg off the PREDICTED heading the
     node turns in place (as the carrot's GoalTurn does) until it is within
     turn_exit_deg, then hands back to MPPI. In the e2e simulator with the Wuhan
     plant (1.23) MPPI alone circled 9.6 turns (corner) and 16 (hedge) without arriving.
  8. PREDICTED VELOCITY. rover_pred gives MPPI the pose 1.3 s ahead, but its velocity
     state came from the measured odometry -- 1.3 s old against that pose. After an
     in-place turn it still read ~0.28 rad/s, MPPI (bound by az_max) planned to keep
     turning, and overshot the route by ~70 deg every time, alternating (e2e corner,
     Wuhan plant: 26 segments, 4.4 turns). This node publishes pred_odom_topic -- the
     predicted pose plus the velocity the command in force will produce -- and MPPI's
     odom_topic points to it. With 6-8 on the Wuhan plant: corner 41 s / 0.4 turns,
     hedge 82 s / 0.9 turns; on the 16-sept plant (0.36): 42 s and 75 s, 0.3 turns.

Never run in the field.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

LATCHED_QOS = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)


def snap_command(v, w, v_floor=0.25, v_stop=0.02, w_floor=0.15, w_stop=0.01,
                 k_w_moving=1.0, k_w_in_place=1.0, w_max=None):
    """MPPI's (v, w) -> what this rover can execute.

    The rover has two usable speed states: stopped, or at least v_floor. So v below
    v_stop is a stop and anything above it is raised to the floor (asking for
    0.15 m/s gets 0 m/s from this rover, not 0.15). v_stop must be SMALL: from rest,
    MPPI's intent to move shows up as ~0.03 m/s (ax_max x model_dt from a measured
    0), and a first version with v_stop 0.10 zeroed exactly that -- the rover never
    moved, odometry stayed 0, MPPI kept asking 0.03 (first e2e run, 180 s).
    w_stop is small for the same reason in place: stuck against the hedge MPPI asked
    0.02-0.05 rad/s (az_max x model_dt from a measured 0), and w_stop 0.05 zeroed it.
    Turning in place, |w| below w_stop is zero and between w_stop and w_floor is
    raised to w_floor. No reverse: negative v is a stop.

    k_w_*: MPPI plans the yaw rate it wants the rover to HAVE; this rover executes
    ~0.36 of the commanded rate while moving and ~1.18 in place (mission_16sept,
    Mission17sept_wuhan 4.1). The command is MPPI's rate divided by that gain, then
    clipped to w_max. Without it MPPI's 0.04 rad/s detour became 0.015 rad/s of
    real turn and the rover drove straight into the hedge (e2e, hedge, local_replan).
    """
    v = float(v)
    w = float(w)
    if v < v_stop:
        v = 0.0
    elif v < v_floor:
        # raising the speed must not open the curve: keep MPPI's curvature w/v, so
        # the rover drives the arc MPPI planned, only faster. Without this a 0.1 m/s
        # arc near a wall became a 2.5x wider one at 0.25 m/s
        w = w * v_floor / v
        v = v_floor
    w = w / (k_w_moving if v > 0.0 else k_w_in_place)
    if w_max is not None:
        w = max(-w_max, min(w_max, w))
    if v == 0.0:
        if abs(w) < w_stop:
            w = 0.0
        elif abs(w) < w_floor:
            w = math.copysign(w_floor, w)
    return v, w


def predict_pose(x, y, yaw, commands, now, delay_s, k_v=1.11, k_w_moving=0.36,
                 k_w_in_place=1.18, v_breakaway=0.15):
    """Roll (x, y, yaw) forward through the commands of the last delay_s.

    commands: [(t, v, w), ...] as sent to /cmd_vel, oldest first; each holds
    until the next. A command sent at t acts on the rover at t + delay_s, so the
    ones sent in (now - delay_s, now] are exactly what will still move the rover
    after `now`: integrating them gives the pose at now + delay_s.
    """
    t0 = now - delay_s
    seg = [(t, v, w) for t, v, w in commands if t > t0]
    before = [c for c in commands if c[0] <= t0]
    if before:
        seg.insert(0, (t0, before[-1][1], before[-1][2]))
    seg.append((now, 0.0, 0.0))
    for (ta, v, w), (tb, _v, _w) in zip(seg[:-1], seg[1:]):
        dt = max(0.0, tb - ta)
        vr = k_v * v if abs(v) >= v_breakaway else 0.0
        wr = (k_w_moving if vr != 0.0 else k_w_in_place) * w
        n = max(1, int(dt / 0.05))
        h = dt / n
        for _ in range(n):
            x += vr * math.cos(yaw) * h
            y += vr * math.sin(yaw) * h
            yaw += wr * h
    return x, y, yaw


def estimate_k_w(commands, yaw_rates, now, delay_s, window_s, v_moving):
    """Median observed/commanded yaw rate over the last window_s, moving and in place.

    commands: [(t, v, w)] sent to the rover (each holds until the next); yaw_rates:
    [(t, w_observed)]. A command sent at t shows at t + delay_s, so each observation
    is paired with the command in force delay_s earlier; only clear turns count
    (|w_cmd| >= 0.1 rad/s, and |w_obs| is not near zero by accident of a transient).
    Returns {'moving': (ratio, n), 'in_place': (ratio, n)}.
    """
    ct = np.array([c[0] for c in commands])
    cv = np.array([c[1] for c in commands])
    cw = np.array([c[2] for c in commands])
    out = {'moving': [], 'in_place': []}
    for t, wo in yaw_rates:
        if t < now - window_s:
            continue
        # the command in force over the 0.5 s before t - delay_s (MPPI changes it every 0.1 s):
        # its mean, and only if it held roughly steady -- a transient pairs badly with the delay
        a = int(np.searchsorted(ct, t - delay_s - 0.5, side='right')) - 1
        b = int(np.searchsorted(ct, t - delay_s, side='right'))
        if a < 0 or b - a < 1:
            continue
        seg_w = cw[a:b]
        wc = float(seg_w.mean())
        vc = float(cv[a:b].min())
        if abs(wc) < 0.1 or float(seg_w.max() - seg_w.min()) > 0.3 * abs(wc):
            continue
        state = 'moving' if vc >= v_moving else ('in_place' if vc == 0.0 else None)
        if state:
            out[state].append(wo / wc)
    return {k: (float(np.median(v)) if v else float('nan'), len(v)) for k, v in out.items()}


def fill_isolated_misses(ranges, range_max, eps=0.05):
    """A bin that reports nothing (range_max) between two bins that both see
    something takes the nearer neighbour's range.

    Nav2's ObstacleLayer clears every cell a ray crosses, so a single bin that
    misses a wall (the profile misses ~3%) raytraces through it and opens a hole
    MPPI can steer into -- it did (e2e, wall). A real opening is wider than one
    5-degree bin at the distances that matter, so it survives this filter.
    """
    r = np.asarray(ranges, float).copy()
    miss = ~np.isfinite(r) | (r >= range_max - eps)
    for i in range(1, len(r) - 1):
        if miss[i] and not miss[i - 1] and not miss[i + 1]:
            r[i] = min(r[i - 1], r[i + 1])
    return r


def densify(points, step=0.1):
    """[(x, y), ...] -> (N, 3) x, y, yaw every `step` m along the polyline."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return np.array([[p[0, 0], p[0, 1], 0.0]]) if len(p) else np.zeros((0, 3))
    out = []
    for a, b in zip(p[:-1], p[1:]):
        d = float(np.hypot(*(b - a)))
        if d < 1e-6:
            continue
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        n = max(1, int(math.ceil(d / step)))
        for i in range(n):
            q = a + (b - a) * (i / n)
            out.append((q[0], q[1], yaw))
    out.append((p[-1, 0], p[-1, 1], out[-1][2] if out else 0.0))
    return np.array(out)


class Nav2RouteFollower(Node):
    def __init__(self):
        super().__init__('nav2_route_follower_node')
        self.declare_parameter('global_route_topic', '/erc/global_route')
        self.declare_parameter('local_route_topic', '/erc/local_route')
        self.declare_parameter('enable_topic', '/enable_inference')
        self.declare_parameter('scan_in_topic', '/erc/free_space_leg')
        self.declare_parameter('scan_out_topic', '/erc/free_space_nav2')
        self.declare_parameter('nav2_cmd_topic', '/nav2/cmd_vel')
        self.declare_parameter('cmd_out_topic', '/omnivla/cmd_vel')
        self.declare_parameter('debug_topic', '/omnivla_debug')
        self.declare_parameter('follow_path_action', 'follow_path')
        self.declare_parameter('clear_costmap_service', '/local_costmap/clear_entirely_local_costmap')
        self.declare_parameter('controller_id', 'FollowPath')
        self.declare_parameter('goal_checker_id', 'goal_checker')
        self.declare_parameter('progress_checker_id', 'progress_checker')
        self.declare_parameter('path_step_m', 0.1)
        self.declare_parameter('crop_deg', 40.0)
        self.declare_parameter('v_floor', 0.25)
        self.declare_parameter('v_stop', 0.02)
        self.declare_parameter('w_floor', 0.15)
        self.declare_parameter('retry_s', 1.0)
        # Measured plant gains (see snap_command); 1.0 disables the compensation. They are
        # PER UNIT: 0.36 moving came from mission_16sept, but the Wuhan units of 22/24-sept
        # executed 1.05-1.26 of the commanded rate moving and 0.89-0.96 in place (gyro vs
        # /cmd_vel). With 0.36 MPPI's turns were multiplied ~3x and the rover drove circles
        # of ~1 m radius for 70 s (mission_24sept_Nav2_circles). Measure the unit before
        # changing these.
        self.declare_parameter('k_w_moving', 1.0)
        self.declare_parameter('k_w_in_place', 1.0)
        self.declare_parameter('w_max', 0.3)     # controller.yaml max_angular_vel
        # Smith predictor (see module docstring, 5); 0 publishes rover_pred = rover_leg
        self.declare_parameter('predict_delay_s', 1.3)
        self.declare_parameter('k_v', 1.11)
        self.declare_parameter('global_frame', 'leg_local')
        self.declare_parameter('base_frame', 'rover_leg')
        self.declare_parameter('pred_frame', 'rover_pred')
        self.declare_parameter('executed_cmd_topic', '/cmd_vel')
        # the velocity MPPI starts from, at the predicted pose's time: what the command in force
        # now will make the rover do (config/nav2_mppi.yaml odom_topic points here). The measured
        # odometry is 1.3 s old against rover_pred: after an in-place turn it still said 0.28 rad/s,
        # and MPPI, bound by its acceleration limits, planned to keep turning -- 70 deg past the
        # route every time (e2e corner, Wuhan plant)
        self.declare_parameter('pred_odom_topic', '/erc/odometry/pred')
        # yaw gain learned online (module docstring, 6); k_w_moving/_in_place are the start values
        self.declare_parameter('k_w_auto', True)
        self.declare_parameter('yaw_rate_topic', '/erc/odometry/local')
        self.declare_parameter('k_w_window_s', 20.0)
        self.declare_parameter('k_w_min_samples', 15)
        self.declare_parameter('k_w_bounds', [0.2, 2.0])
        # turn in place first (module docstring, 7); 0 disables
        self.declare_parameter('turn_enter_deg', 50.0)
        self.declare_parameter('turn_exit_deg', 20.0)
        self.declare_parameter('turn_rate', 0.3)          # rad/s the rover should HAVE in place
        self.declare_parameter('turn_lookahead_m', 1.0)
        p = self.get_parameter

        # nav2_msgs is only needed here; imported late so the error says what is missing
        try:
            from nav2_msgs.action import FollowPath
            from nav2_msgs.srv import ClearEntireCostmap
        except ImportError as exc:
            raise SystemExit(f'nav2_msgs not found ({exc}). Install Nav2: '
                             'sudo apt install ros-jazzy-nav2-controller ros-jazzy-nav2-mppi-controller '
                             'ros-jazzy-nav2-costmap-2d ros-jazzy-nav2-lifecycle-manager ros-jazzy-nav2-msgs')
        self._FollowPath = FollowPath
        self._client = ActionClient(self, FollowPath, p('follow_path_action').value)
        self._clear = self.create_client(ClearEntireCostmap, p('clear_costmap_service').value)
        self._ClearReq = ClearEntireCostmap.Request

        self._route = None            # (frame_id, points) currently wanted
        self._route_key = None
        self._sent_key = None
        self._enabled = False
        self._goal_handle = None
        self._retry_at = None

        self.create_subscription(Path, p('global_route_topic').value, self._on_global, LATCHED_QOS)
        self.create_subscription(Path, p('local_route_topic').value, self._on_local, LATCHED_QOS)
        self.create_subscription(Bool, p('enable_topic').value, self._on_enable, LATCHED_QOS)
        self.create_subscription(LaserScan, p('scan_in_topic').value, self._on_scan, 10)
        self.create_subscription(Twist, p('nav2_cmd_topic').value, self._on_cmd, 10)
        self._scan_pub = self.create_publisher(LaserScan, p('scan_out_topic').value, 10)
        self._cmd_pub = self.create_publisher(Twist, p('cmd_out_topic').value, 10)
        self._dbg_pub = self.create_publisher(String, p('debug_topic').value, 10)
        self.create_timer(0.2, self._tick)
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)
        self._tf_pub = TransformBroadcaster(self)
        self._odom_pub = self.create_publisher(Odometry, p('pred_odom_topic').value, 10)
        self._sent = []                 # (t, v, w) actually sent to the rover
        self._yaw_rates = []            # (t, observed yaw rate)
        self._k_w = {'moving': float(p('k_w_moving').value), 'in_place': float(p('k_w_in_place').value)}
        self._dense = None              # (frame, (N, 3)) the path last sent to MPPI
        self._pred = None               # (x, y, yaw) last predicted pose
        self._turning = 0.0             # sign of the in-place turn in progress, 0 = MPPI drives
        self.create_subscription(Odometry, p('yaw_rate_topic').value, self._on_yaw_rate, 10)
        self.create_timer(1.0, self._update_k_w)
        self.create_subscription(Twist, p('executed_cmd_topic').value, self._on_executed, 10)
        self.create_timer(0.05, self._publish_prediction)
        self.get_logger().info('Nav2 route follower ready: route -> FollowPath (MPPI), '
                               f'{p("scan_in_topic").value} -> {p("scan_out_topic").value}, '
                               f'{p("nav2_cmd_topic").value} -> {p("cmd_out_topic").value}')

    # -- inputs -----------------------------------------------------------
    def _set_route(self, msg, kind):
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        if len(pts) < 2:
            return
        key = (kind, msg.header.stamp.sec, msg.header.stamp.nanosec, len(pts))
        self._route = (msg.header.frame_id or 'leg_local', pts)
        self._route_key = key
        self.get_logger().info(f'{kind} route: {len(pts)} points, '
                               f'{sum(math.dist(a, b) for a, b in zip(pts, pts[1:])):.1f} m')

    def _on_global(self, msg):
        # a new leg: its frame restarts at the rover, so the old marks are in the wrong place
        self._clear_costmap()
        self._set_route(msg, 'global')

    def _on_local(self, msg):
        self._set_route(msg, 'local')

    def _on_enable(self, msg):
        if msg.data != self._enabled:
            self.get_logger().info(f'enable_inference -> {msg.data}')
        self._enabled = bool(msg.data)
        if not self._enabled:
            self._cancel()

    def _on_scan(self, msg):
        crop = math.radians(float(self.get_parameter('crop_deg').value))
        n = len(msg.ranges)
        ang = msg.angle_min + np.arange(n) * msg.angle_increment
        keep = np.where(np.abs(ang) <= crop + 1e-6)[0]
        if len(keep) == 0:
            return
        out = LaserScan()
        out.header = msg.header
        out.angle_min = float(ang[keep[0]])
        out.angle_max = float(ang[keep[-1]])
        out.angle_increment = msg.angle_increment
        out.range_min, out.range_max = msg.range_min, msg.range_max
        out.ranges = [float(x) for x in fill_isolated_misses([msg.ranges[i] for i in keep], msg.range_max)]
        self._scan_pub.publish(out)

    def _on_cmd(self, msg):
        g = lambda n: float(self.get_parameter(n).value)  # noqa: E731
        vin, win, source = msg.linear.x, msg.angular.z, 'mppi'
        err = self._heading_error()
        if err is not None and g('turn_enter_deg') > 0.0:
            if self._turning == 0.0 and abs(err) > math.radians(g('turn_enter_deg')):
                self._turning = math.copysign(1.0, err)
            elif self._turning != 0.0 and abs(err) < math.radians(g('turn_exit_deg')):
                self._turning = 0.0
        else:
            self._turning = 0.0
        if self._turning != 0.0:
            vin, win, source = 0.0, self._turning * g('turn_rate'), 'turn-in-place'
        v, w = snap_command(vin, win, g('v_floor'), g('v_stop'), g('w_floor'),
                            k_w_moving=self._k_w['moving'], k_w_in_place=self._k_w['in_place'],
                            w_max=g('w_max'))
        if not self._enabled:
            v, w = 0.0, 0.0
        out = Twist()
        out.linear.x, out.angular.z = v, w
        self._cmd_pub.publish(out)
        self._dbg_pub.publish(String(data=(
            f'[mppi] v_raw={msg.linear.x:.3f} w_raw={msg.angular.z:.3f} | '
            f'linear_cmd={v:.4f} angular_cmd={w:.4f} | source={source} | '
            f'k_w moving={self._k_w["moving"]:.2f} in_place={self._k_w["in_place"]:.2f}'
            + (f' | route_err={math.degrees(err):+.0f}deg' if err is not None else ''))))

    # -- the predicted pose -------------------------------------------------
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_executed(self, msg):
        now = self._now()
        self._sent.append((now, msg.linear.x, msg.angular.z))
        keep = now - 3.0 - float(self.get_parameter('predict_delay_s').value)
        while len(self._sent) > 2 and self._sent[1][0] < keep:
            self._sent.pop(0)

    def _publish_prediction(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        try:
            tr = self._tf_buf.lookup_transform(g('global_frame'), g('base_frame'), rclpy.time.Time())
        except Exception:
            return
        q = tr.transform.rotation
        x, y = tr.transform.translation.x, tr.transform.translation.y
        yaw = 2.0 * math.atan2(q.z, q.w)
        d = float(g('predict_delay_s'))
        if d > 0.0:
            x, y, yaw = predict_pose(x, y, yaw, self._sent, self._now(), d, float(g('k_v')),
                                     self._k_w['moving'], self._k_w['in_place'])
        self._pred = (x, y, yaw)
        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id, out.child_frame_id = g('global_frame'), g('pred_frame')
        out.transform.translation.x, out.transform.translation.y = float(x), float(y)
        out.transform.rotation.z, out.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self._tf_pub.sendTransform(out)
        od = Odometry()
        od.header.stamp = out.header.stamp
        od.header.frame_id, od.child_frame_id = g('global_frame'), g('pred_frame')
        od.pose.pose.position.x, od.pose.pose.position.y = float(x), float(y)
        od.pose.pose.orientation = out.transform.rotation
        if self._sent:
            _t, vc, wc = self._sent[-1]
            vr = float(g('k_v')) * vc if abs(vc) >= 0.15 else 0.0
            od.twist.twist.linear.x = float(vr)
            od.twist.twist.angular.z = float((self._k_w['moving'] if vr else self._k_w['in_place']) * wc)
        self._odom_pub.publish(od)

    # -- learned yaw gain and turn-in-place ---------------------------------
    def _on_yaw_rate(self, msg):
        now = self._now()
        self._yaw_rates.append((now, float(msg.twist.twist.angular.z)))
        keep = now - float(self.get_parameter('k_w_window_s').value) - 5.0
        while self._yaw_rates and self._yaw_rates[0][0] < keep:
            self._yaw_rates.pop(0)

    def _update_k_w(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        if not g('k_w_auto') or len(self._sent) < 2 or not self._yaw_rates:
            return
        k = estimate_k_w(self._sent, self._yaw_rates, self._now(), float(g('predict_delay_s')),
                         float(g('k_w_window_s')), float(g('v_floor')) * 0.8)
        lo, hi = [float(b) for b in g('k_w_bounds')]
        for state, (ratio, n) in k.items():
            if n >= int(g('k_w_min_samples')) and np.isfinite(ratio):
                new = min(hi, max(lo, ratio))
                if abs(new - self._k_w[state]) > 0.1:
                    self.get_logger().info(f'yaw gain {state}: {self._k_w[state]:.2f} -> {new:.2f} '
                                           f'(median of {n} samples)')
                self._k_w[state] = new

    def _heading_error(self):
        """Signed angle from the PREDICTED heading to the path point turn_lookahead_m ahead."""
        if self._pred is None or self._dense is None:
            return None
        _frame, d = self._dense
        x, y, yaw = self._pred
        i = int(np.argmin(np.hypot(d[:, 0] - x, d[:, 1] - y)))
        j = min(len(d) - 1, i + max(1, int(float(self.get_parameter('turn_lookahead_m').value)
                                           / float(self.get_parameter('path_step_m').value))))
        if math.hypot(d[j, 0] - x, d[j, 1] - y) < 0.3:
            return None               # at the end of the path: nothing to face
        a = math.atan2(d[j, 1] - y, d[j, 0] - x) - yaw
        return (a + math.pi) % (2 * math.pi) - math.pi

    # -- the goal ---------------------------------------------------------
    def _tick(self):
        if not self._enabled or self._route is None:
            return
        if self._sent_key == self._route_key and self._goal_handle is not None:
            return
        if self._retry_at is not None and self.get_clock().now().nanoseconds * 1e-9 < self._retry_at:
            return
        if not self._client.server_is_ready():
            self.get_logger().warn('controller_server follow_path not available yet',
                                   throttle_duration_sec=5)
            return
        self._send()

    def _send(self):
        frame, pts = self._route
        dense = densify(pts, float(self.get_parameter('path_step_m').value))
        # Start the path where the rover is. MPPI looks for the rover only within
        # prune_distance of the path's start: a leg's full route re-sent mid-leg (after
        # an abort, or back from a local detour) put the rover beyond that, the part MPPI
        # did look at fell outside the 8 m rolling costmap, and every goal aborted with
        # "Resulting plan has 0 poses" -- ~180 times in e2e wall and chicane.
        try:
            tr = self._tf_buf.lookup_transform(frame, self.get_parameter('base_frame').value,
                                               rclpy.time.Time())
            rx, ry = tr.transform.translation.x, tr.transform.translation.y
            i = int(np.argmin(np.hypot(dense[:, 0] - rx, dense[:, 1] - ry)))
            dense = dense[i:] if len(dense) - i >= 2 else dense[-2:]
        except Exception:
            pass
        self._dense = (frame, dense)
        path = Path()
        path.header.frame_id = frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y, yaw in dense:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            path.poses.append(ps)
        goal = self._FollowPath.Goal()
        goal.path = path
        goal.controller_id = self.get_parameter('controller_id').value
        goal.goal_checker_id = self.get_parameter('goal_checker_id').value
        goal.progress_checker_id = self.get_parameter('progress_checker_id').value
        key = self._route_key
        self._sent_key = key
        self.get_logger().info(f'FollowPath: {len(path.poses)} poses in {frame}')
        fut = self._client.send_goal_async(goal)
        fut.add_done_callback(lambda f, k=key: self._on_goal(f, k))

    def _on_goal(self, fut, key):
        handle = fut.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn('FollowPath goal rejected; retrying')
            self._sent_key = None
            self._retry_at = self.get_clock().now().nanoseconds * 1e-9 + float(self.get_parameter('retry_s').value)
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f, k=key: self._on_result(f, k))

    def _on_result(self, fut, key):
        res = fut.result()
        status = res.status if res is not None else -1
        if key == self._sent_key:
            self._goal_handle = None
        # 4 SUCCEEDED, 5 CANCELED, 6 ABORTED
        if status == 6 and self._enabled and key == self._sent_key:
            self.get_logger().warn('FollowPath aborted (MPPI failed or no progress); resending')
            self._sent_key = None
            self._retry_at = self.get_clock().now().nanoseconds * 1e-9 + float(self.get_parameter('retry_s').value)
        elif status == 4:
            self.get_logger().info('FollowPath succeeded: end of route')

    def _cancel(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None
        self._sent_key = None
        self._cmd_pub.publish(Twist())

    def _clear_costmap(self):
        if self._clear.service_is_ready():
            self._clear.call_async(self._ClearReq())

    def destroy_node(self):
        if self.context.ok():
            self._cmd_pub.publish(Twist())
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Nav2RouteFollower()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
