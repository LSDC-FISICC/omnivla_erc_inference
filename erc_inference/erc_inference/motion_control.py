"""Motion control laws for omnivla_edge_node and checkpoint_controller_node.

Kept free of ROS and torch so test/controller_sim.py drives the rover model with
exactly the code that runs on the robot.
"""

import math
from collections import deque

import numpy as np


CONTROLLER_TYPES = ("polar", "pid")
STEERING_SOURCES = ("carrot", "model")
LINEAR_LAWS = ("cruise", "rho")


def clip_angle(angle: float) -> float:
    """Wrap an angle (rad) to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def robot_frame_offset(delta_x, delta_y, heading_deg):
    """World offset (east, north) -> (right, forward) for a heading in SDK degrees.

    SDK convention: 0 = North, clockwise-positive. This is the rotation the model's
    goal_pose token was built with, so it must not change.
    """
    heading_rad = -float(heading_deg) / 180.0 * math.pi
    rel_x = delta_x * math.cos(heading_rad) + delta_y * math.sin(heading_rad)
    rel_y = -delta_x * math.sin(heading_rad) + delta_y * math.cos(heading_rad)
    return rel_x, rel_y


def goal_bearing_from_offset(rel_x, rel_y):
    """Bearing (rad, left-positive) of a (right, forward) offset."""
    return math.atan2(-rel_x, rel_y)


class HeadingPID:
    """PID on the bearing error to the selected waypoint.

    Tuned against mission9sept, where the previous law (`arctan(dy/dx) / DT`
    with DT = 1/3 s) multiplied the bearing by 3 and therefore saturated the
    angular command at a bearing of only 5.73 deg. 80.6% of the ticks in that
    run asked for more than max_angular_vel, which made the output effectively
    three-valued -- hard left, hard right, straight. The default kp of 0.4
    instead saturates at ~43 deg, leaving a real proportional band. See the
    pid.kp parameter for how the gains were chosen.

    Anti-windup matters more than usual here: the rover does not always execute
    what it is told (the ratio of achieved to commanded yaw rate measured 0.98
    on gentle turns at cruise but 0.29 while slowed mid-turn), so a naive
    integrator would wind up against a deficit no amount of integral can fix.
    Integration is therefore frozen whenever the output is saturated and the
    error would push it further into the stop.

    Not thread-safe, and does not need to be: this node runs on a plain
    rclpy.spin() single-threaded executor, so step() (timer) and reset()
    (subscription callbacks) never overlap. That stops being true if anyone
    moves it to a MultiThreadedExecutor, as checkpoint_controller_node uses.
    """

    # The timer does not tick evenly. Measured on mission9sept: 7% of the 396
    # real ticks arrived less than 10 ms after the previous one (minimum 2 us,
    # the timer catching up after a slow forward pass) and one gap reached
    # 1.68 s. Dividing by a 2 us dt amplifies the derivative by ~500,000x --
    # in an early version of this class that produced a D term of 75 rad/s
    # against an output limit of 0.3. Both ends are clamped: a burst tick
    # reuses roughly the nominal period rather than exploding, and a long
    # stall does not dump a huge slab into the integral.
    DT_MIN = 0.05
    DT_MAX = 1.0

    def __init__(self, kp, ki, kd, out_limit, integral_limit, derivative_alpha):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.integral_limit = integral_limit
        self.derivative_alpha = derivative_alpha
        self.reset()

    def reset(self):
        """Clear history. Call on enable/disable edges and on a new goal.

        Without this, the integral accumulated chasing the previous waypoint is
        applied to the next one, and the step in bearing when a new goal
        arrives produces a derivative kick.
        """
        self._integral = 0.0
        self._prev_error = None
        self._deriv = 0.0

    def step(self, error, dt):
        """error: bearing to the waypoint (rad). dt: measured tick period (s)."""
        dt = float(np.clip(dt, self.DT_MIN, self.DT_MAX))

        proportional = self.kp * error

        if self._prev_error is None:
            raw_deriv = 0.0
        else:
            raw_deriv = (error - self._prev_error) / dt
        # The bearing comes from a fresh forward pass every tick and jitters, so
        # an unfiltered derivative at ~3 Hz is mostly noise.
        self._deriv += self.derivative_alpha * (raw_deriv - self._deriv)
        self._prev_error = error
        # Belt and braces on top of DT_MIN: however the derivative was arrived
        # at, it may not on its own command more than the output limit.
        derivative = float(np.clip(self.kd * self._deriv,
                                   -self.out_limit, self.out_limit))

        unsaturated = proportional + self.ki * self._integral + derivative
        # Conditional integration: only accumulate when that would not drive an
        # already-saturated output further into its limit.
        saturated_high = unsaturated >= self.out_limit and error > 0.0
        saturated_low = unsaturated <= -self.out_limit and error < 0.0
        if not (saturated_high or saturated_low):
            self._integral = float(
                np.clip(self._integral + error * dt,
                        -self.integral_limit, self.integral_limit)
            )

        output = proportional + self.ki * self._integral + derivative
        return (float(np.clip(output, -self.out_limit, self.out_limit)),
                float(proportional), float(self.ki * self._integral),
                float(derivative))


def polar_control(dx, dy, hx, hy, k_rho, k_alpha, k_beta, max_linear, max_angular,
                  backward_allowed=False):
    """Siegwart's polar-coordinate controller, applied to one predicted waypoint.

    Siegwart & Nourbakhsh, "Introduction to Autonomous Mobile Robots", 3.6.2.
    Everything is in the robot frame the model predicts in: the robot sits at
    the origin with heading 0, and the goal pose is the selected waypoint --
    position (dx, dy) and heading atan2(hy, hx).

        rho   = distance to the waypoint
        alpha = bearing of the waypoint relative to the robot's heading
        beta  = heading the model wants at the waypoint, minus that bearing
        v     = k_rho * rho
        w     = k_alpha * alpha + k_beta * beta

    Only used with polar.linear_law "rho" and polar.steering_source "model", i.e.
    the controller as it drove mission_16sept. There v = k_rho * rho fell to
    0.05 m/s whenever the model predicted a short waypoint, below the speed at
    which the rover moves at all; "cruise" (cruise_speed) replaces it.

    Deliberate differences from a straight port of the usual implementation:
    - w is clipped symmetrically to +-max_angular.
    - v is clipped to max_linear; k_rho * rho has no bound of its own.
    - beta is wrapped to [-pi, pi] like alpha.
    - Driving backward redefines the robot's forward axis (alpha += pi, v < 0).
    - No "goal reached, stop" latch: arrival is decided by
      checkpoint_controller_node against GPS.

    Returns (v, w, rho, alpha, beta), angles in rad.
    """
    rho = float(math.hypot(dx, dy))
    goal_heading = math.atan2(hy, hx)
    if rho < 1e-6:
        # The model predicts no displacement: turn in place toward the heading
        # it predicts. alpha is undefined at rho = 0, and with k_beta < 0 the
        # beta term alone would turn AWAY from that heading.
        w = float(np.clip(k_alpha * goal_heading, -max_angular, max_angular))
        return 0.0, w, 0.0, 0.0, goal_heading

    bearing = math.atan2(dy, dx)
    alpha = clip_angle(bearing)
    beta = clip_angle(goal_heading - bearing)

    direction = 1.0
    if backward_allowed and abs(alpha) > math.pi / 2:
        direction = -1.0
        alpha = clip_angle(alpha + math.pi)

    v_min = -max_linear if backward_allowed else 0.0
    v = float(np.clip(direction * k_rho * rho, v_min, max_linear))
    w = float(np.clip(k_alpha * alpha + k_beta * beta, -max_angular, max_angular))
    return v, w, rho, alpha, beta


def polar_gain_warnings(k_rho, k_alpha, k_beta):
    """Siegwart's stability conditions for polar_control, as warning strings."""
    warnings = []
    if k_rho <= 0.0:
        warnings.append(f"k_rho={k_rho} must be > 0")
    # Siegwart needs k_beta < 0 to converge on a final heading. The waypoint
    # here is a moving target ~1.5 s ahead, not a pose to arrive at, so 0
    # (bearing tracking only) is legitimate -- and the default.
    if k_beta > 0.0:
        warnings.append(f"k_beta={k_beta} must be <= 0 (0 = bearing tracking only)")
    if k_alpha - k_rho <= 0.0:
        warnings.append(f"k_alpha - k_rho = {k_alpha - k_rho:.3f} must be > 0")
    strong = k_alpha + (5.0 / 3.0) * k_beta - (2.0 / math.pi) * k_rho
    if strong <= 0.0:
        warnings.append(f"k_alpha + 5/3*k_beta - 2/pi*k_rho = {strong:.3f} <= 0: "
                        "the rover may reverse its direction of travel mid-approach")
    return warnings


def cruise_speed(bearing, max_linear, min_linear, slowdown_start_rad, slowdown_end_rad):
    """Forward speed for a bearing error: max_linear, tapering to min_linear.

    Never returns a value in (0, min_linear). On mission_16sept every command
    held at 0.049-0.065 m/s left the rover standing still while every command
    held at ~0.3 m/s moved it, so a speed between zero and the rover's breakaway
    speed is a stop that the controller does not know it asked for.
    """
    magnitude = abs(bearing)
    if magnitude <= slowdown_start_rad:
        return float(max_linear)
    if magnitude >= slowdown_end_rad:
        return float(min_linear)
    t = (magnitude - slowdown_start_rad) / max(slowdown_end_rad - slowdown_start_rad, 1e-9)
    return float(max_linear + t * (min_linear - max_linear))


def steer(bearing, k_heading, max_angular, deadzone_rad):
    """Proportional yaw-rate command with a continuous deadzone around zero.

    The deadzone keeps GPS/heading jitter (a 0.2 m position error against a
    carrot 1.5 m ahead is ~8 deg) from turning a straight run into a weave.
    Shifting rather than zeroing keeps the command continuous at its edge.
    """
    magnitude = abs(bearing) - deadzone_rad
    if magnitude <= 0.0:
        return 0.0
    return float(np.clip(math.copysign(k_heading * magnitude, bearing), -max_angular, max_angular))


class DelayCompensator:
    """Heading change still on its way from yaw-rate commands already sent.

    Commands take T_d ~ 1.3 s to act (Tarea B), so a bearing measured now does not
    yet show the turn the last 1.3 s of commands will produce. Subtracting that
    turn from the bearing is a Smith predictor reduced to its yaw channel.
    `gain` is the yaw rate achieved per unit commanded; 0 disables the correction.
    """

    def __init__(self):
        self._history = deque()

    def reset(self):
        self._history.clear()

    def record(self, now, angular_cmd):
        self._history.append((float(now), float(angular_cmd)))

    def pending_turn(self, now, delay_s, gain):
        """Integral of gain * angular_cmd over (now - delay_s, now], in rad."""
        if delay_s <= 0.0 or gain == 0.0 or not self._history:
            return 0.0
        start = now - delay_s
        while len(self._history) > 1 and self._history[1][0] <= start:
            self._history.popleft()
        turn = 0.0
        items = list(self._history)
        for i, (t, w) in enumerate(items):
            t_next = items[i + 1][0] if i + 1 < len(items) else now
            lo, hi = max(t, start), min(t_next, now)
            if hi > lo:
                turn += w * (hi - lo)
        return gain * turn


class GoalTurn:
    """Turn toward the goal when it is behind the rover.

    The model's selected waypoint always sits in the forward half-plane -- on
    mission_10sept its bearing never went beyond +-18 deg -- so a leg that starts
    with the goal behind would otherwise drive away from it.

    Driven by localization, not by the model. Hysteresis keeps it from
    chattering: engage past enter_deg, release under exit_deg. The rover keeps
    rotating for ~1.3 s after the command drops, which is why exit_deg is well
    short of zero.

    brake_s: after engaging, command a full stop for this long before turning.
    On mission_16sept the rover went from 0.22 m/s forward to a 0.3 rad/s spin in
    one tick, and was pitched 41 deg into that spin two seconds later (cause not
    established; the step is what this removes).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.direction = 0.0
        self.engaged_at = None

    def update(self, bearing, distance, enabled, enter_deg, exit_deg, min_distance_m, now=0.0):
        """bearing: rad to the goal, left-positive. Returns whether to override."""
        if not enabled or distance < min_distance_m:
            self.reset()
            return False
        magnitude = abs(math.degrees(bearing))
        if self.active:
            if magnitude < exit_deg:
                self.reset()
        elif magnitude > enter_deg:
            self.active = True
            self.engaged_at = float(now)
            # Latched when engaging: near 180 deg the sign of the bearing flips
            # with every bit of heading noise, and re-reading it each tick would
            # reverse the turn halfway through.
            self.direction = 1.0 if bearing >= 0.0 else -1.0
        return self.active

    def braking(self, now, brake_s):
        return self.active and self.engaged_at is not None and now - self.engaged_at < brake_s


class SlewLimiter:
    """Rate limit on one velocity channel: `accel` away from zero, `decel` toward it.

    A sign change is taken through zero at `decel` first. Rates are per second;
    a non-positive rate disables limiting in that direction.
    """

    def __init__(self, accel, decel):
        self.accel = accel
        self.decel = decel
        self.value = 0.0

    def reset(self, value=0.0):
        self.value = float(value)

    def step(self, target, dt):
        target = float(target)
        current = self.value
        dt = max(float(dt), 0.0)
        if current != 0.0 and (target == 0.0 or math.copysign(1.0, target) != math.copysign(1.0, current)):
            # Heading toward zero first.
            limit = self.decel * dt if self.decel > 0.0 else math.inf
            if abs(current) <= limit:
                remaining = dt - (abs(current) / self.decel if self.decel > 0.0 else 0.0)
                self.value = 0.0
                if target != 0.0 and remaining > 0.0:
                    return self.step(target, remaining)
                return self.value
            self.value = current - math.copysign(limit, current)
            return self.value
        if abs(target) > abs(current):
            limit = self.accel * dt if self.accel > 0.0 else math.inf
        else:
            limit = self.decel * dt if self.decel > 0.0 else math.inf
        delta = target - current
        self.value = current + math.copysign(min(abs(delta), limit), delta)
        return self.value


# Seconds between consecutive waypoints inside the model's predicted chunk.
# LogoNav trains with action_spacing=3 on 10 Hz data, so the 8 waypoints of the
# chunk sit at +0.3 s ... +2.4 s. This is NOT the controller tick period.
WAYPOINT_DT = 0.3

# Every parameter MotionController reads, with its default. omnivla_edge_node
# declares its ROS parameters from this table and config/controller.yaml
# overrides it, so the node and test/controller_sim.py cannot drift apart.
# Rationale for each value lives in config/controller.yaml.
DEFAULT_PARAMS = {
    "tick_rate": 3.0,
    "max_linear_vel": 0.3,
    "max_angular_vel": 0.3,
    "controller_type": "polar",
    "pid.kp": 0.4,
    "pid.ki": 0.03,
    "pid.kd": 0.10,
    "pid.integral_limit": 0.67,
    "pid.derivative_alpha": 0.4,
    "pid.turn_slowdown_start_deg": 60.0,
    "pid.turn_slowdown_end_deg": 120.0,
    "pid.turn_speed_floor": 0.4,
    "polar.steering_source": "carrot",
    "polar.linear_law": "cruise",
    "polar.k_rho": 0.5,
    "polar.k_alpha": 1.5,
    "polar.k_beta": 0.0,
    "polar.backward_allowed": False,
    "polar.k_heading": 0.5,
    "polar.min_linear_vel": 0.25,
    "polar.slowdown_start_deg": 30.0,
    "polar.slowdown_end_deg": 60.0,
    "polar.heading_deadzone_deg": 6.0,
    "polar.delay_compensation_s": 1.3,
    "polar.delay_compensation_gain": 1.0,
    "goal_turn.enabled": True,
    "goal_turn.enter_deg": 90.0,
    "goal_turn.exit_deg": 45.0,
    "goal_turn.angular_vel": 0.3,
    "goal_turn.linear_vel": 0.0,
    "goal_turn.min_distance_m": 1.0,
    "goal_turn.brake_s": 1.0,
}


def parameter_errors(params):
    """Reasons a parameter set cannot drive the rover; empty when it can."""
    errors = []
    if params["controller_type"] not in CONTROLLER_TYPES:
        errors.append(f"controller_type must be one of {CONTROLLER_TYPES}, got {params['controller_type']!r}")
    if params["polar.steering_source"] not in STEERING_SOURCES:
        errors.append(f"polar.steering_source must be one of {STEERING_SOURCES}, "
                      f"got {params['polar.steering_source']!r}")
    if params["polar.linear_law"] not in LINEAR_LAWS:
        errors.append(f"polar.linear_law must be one of {LINEAR_LAWS}, got {params['polar.linear_law']!r}")
    if not 0.0 <= params["polar.min_linear_vel"] <= params["max_linear_vel"]:
        errors.append("polar.min_linear_vel must lie in [0, max_linear_vel]")
    if params["polar.slowdown_end_deg"] < params["polar.slowdown_start_deg"]:
        errors.append("polar.slowdown_end_deg must be >= polar.slowdown_start_deg")
    return errors


class MotionController:
    """Turns localization + the model's waypoint into (linear, angular) for one tick."""

    def __init__(self, params):
        self.heading_pid = HeadingPID(
            kp=params["pid.kp"], ki=params["pid.ki"], kd=params["pid.kd"],
            out_limit=params["max_angular_vel"], integral_limit=params["pid.integral_limit"],
            derivative_alpha=params["pid.derivative_alpha"])
        self.goal_turn = GoalTurn()
        self.delay_compensator = DelayCompensator()
        self._last_tick_time = None
        self._last_controller_type = None

    def reset(self):
        """Clear all history: enable/disable edges and new targets."""
        self.heading_pid.reset()
        self.goal_turn.reset()
        self.delay_compensator.reset()
        self._last_tick_time = None

    def note_idle(self, now):
        """The node published a stop without running the controller."""
        self.delay_compensator.record(now, 0.0)

    def command(self, now, params, goal_bearing, goal_distance, waypoint, waypoint_select, use_pose_goal):
        """Returns (mode, linear, angular, detail).

        goal_bearing (rad, left-positive) and goal_distance (m) locate the goal
        the checkpoint controller streams (the carrot) from localization alone.
        waypoint is the model's selected (dx, dy, hx, hy) in meters.
        """
        max_linear = params["max_linear_vel"]
        max_angular = params["max_angular_vel"]
        controller_type = params["controller_type"]

        was_turning = self.goal_turn.active
        if use_pose_goal:
            turning = self.goal_turn.update(
                goal_bearing, float(goal_distance),
                enabled=params["goal_turn.enabled"],
                enter_deg=params["goal_turn.enter_deg"],
                exit_deg=params["goal_turn.exit_deg"],
                min_distance_m=params["goal_turn.min_distance_m"],
                now=now)
        else:
            self.goal_turn.reset()
            turning = False
        if was_turning and not turning:
            self.heading_pid.reset()
            self._last_tick_time = None

        if controller_type != self._last_controller_type:
            # Switching laws mid-drive: the PID's history describes the other law.
            self.heading_pid.reset()
            self._last_tick_time = None
            self._last_controller_type = controller_type

        if turning:
            mode = "goal-turn"
            if self.goal_turn.braking(now, params["goal_turn.brake_s"]):
                linear, angular, detail = 0.0, 0.0, "braking before the turn"
            else:
                linear = float(np.clip(params["goal_turn.linear_vel"], 0.0, max_linear))
                angular = self.goal_turn.direction * min(abs(params["goal_turn.angular_vel"]), max_angular)
                detail = "controller bypassed"
        elif controller_type == "polar":
            mode = "polar"
            linear, angular, detail = self._polar(now, params, goal_bearing, waypoint, max_linear, max_angular)
        elif controller_type == "pid":
            mode = "pid"
            linear, angular, detail = self._pid(now, params, waypoint, waypoint_select, max_linear, max_angular)
        else:
            raise ValueError(f"unknown controller_type {controller_type!r}")

        self.delay_compensator.record(now, angular)
        return mode, float(linear), float(angular), detail

    def _polar(self, now, params, goal_bearing, waypoint, max_linear, max_angular):
        dx, dy, hx, hy = waypoint
        source = params["polar.steering_source"]
        if source == "model" and params["polar.linear_law"] == "rho":
            v, w, rho, alpha, beta = polar_control(
                dx, dy, hx, hy,
                k_rho=params["polar.k_rho"], k_alpha=params["polar.k_alpha"], k_beta=params["polar.k_beta"],
                max_linear=max_linear, max_angular=max_angular,
                backward_allowed=params["polar.backward_allowed"])
            return v, w, (f"rho={rho:.3f}m alpha={math.degrees(alpha):+.1f}deg "
                          f"beta={math.degrees(beta):+.1f}deg")

        model_alpha = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else math.atan2(hy, hx)
        if source == "carrot":
            pending = self.delay_compensator.pending_turn(
                now, params["polar.delay_compensation_s"], params["polar.delay_compensation_gain"])
            bearing = clip_angle(goal_bearing - pending)
            k = params["polar.k_heading"]
        else:
            pending = 0.0
            bearing = model_alpha
            k = params["polar.k_alpha"]

        angular = steer(bearing, k, max_angular, math.radians(params["polar.heading_deadzone_deg"]))
        if params["polar.linear_law"] == "rho":
            linear = float(np.clip(params["polar.k_rho"] * math.hypot(dx, dy), 0.0, max_linear))
        else:
            linear = cruise_speed(bearing, max_linear, params["polar.min_linear_vel"],
                                  math.radians(params["polar.slowdown_start_deg"]),
                                  math.radians(params["polar.slowdown_end_deg"]))
        return linear, angular, (f"steer={source} bearing={math.degrees(bearing):+.1f}deg "
                                 f"pending={math.degrees(pending):+.1f}deg "
                                 f"model_alpha={math.degrees(model_alpha):+.1f}deg rho={math.hypot(dx, dy):.3f}m")

    def _pid(self, now, params, waypoint, waypoint_select, max_linear, max_angular):
        dx, dy, hx, hy = waypoint
        eps = 1e-8
        # Gains are re-read every tick so they can be tuned live with `ros2 param set`.
        pid = self.heading_pid
        pid.kp, pid.ki, pid.kd = params["pid.kp"], params["pid.ki"], params["pid.kd"]
        pid.integral_limit = params["pid.integral_limit"]
        pid.derivative_alpha = params["pid.derivative_alpha"]
        pid.out_limit = max_angular

        # Measured tick period, not the nominal one.
        dt = 1.0 / params["tick_rate"] if self._last_tick_time is None else now - self._last_tick_time
        self._last_tick_time = now

        # atan2, not atan(dy/dx), so a waypoint behind the rover reads near +-pi.
        if abs(dx) < eps and abs(dy) < eps:
            heading_error = clip_angle(math.atan2(hy, hx))
            distance = 0.0
        else:
            heading_error = clip_angle(math.atan2(dy, dx))
            distance = float(math.hypot(dx, dy))

        angular, p_term, i_term, d_term = pid.step(heading_error, dt)

        # The chunk says where the rover should be at +(waypoint_select + 1) *
        # WAYPOINT_DT seconds, so the speed it implies is distance / that horizon.
        linear_raw = distance / ((waypoint_select + 1) * WAYPOINT_DT)

        # Full speed while the goal is roughly ahead, tapering to turn_speed_floor
        # once it is far enough off-axis that driving forward no longer closes it.
        slow_start = math.radians(params["pid.turn_slowdown_start_deg"])
        slow_end = math.radians(params["pid.turn_slowdown_end_deg"])
        floor = params["pid.turn_speed_floor"]
        abs_err = abs(heading_error)
        if abs_err <= slow_start:
            scale = 1.0
        elif abs_err >= slow_end:
            scale = floor
        else:
            t = (abs_err - slow_start) / max(slow_end - slow_start, eps)
            scale = 1.0 + t * (floor - 1.0)

        linear = float(np.clip(linear_raw * scale, 0.0, max_linear))
        detail = (f"heading_err={math.degrees(heading_error):+.1f}deg dist={distance:.3f}m "
                  f"dt={dt:.3f}s | P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f} | "
                  f"linear_raw={linear_raw:.4f} scale={scale:.2f}")
        return linear, angular, detail


class CommandShaper:
    """Acceleration limits on both channels of the command sent to the rover.

    On mission_16sept the command stepped from forward motion straight into a
    0.3 rad/s spin, and from that spin straight to 0.3 m/s, each within one
    tick. The rover had pitched to 41 deg in the first of those spins and rolled
    over after the second (cause not established). Shaping every command on the
    way out removes those steps whichever law produced them.
    """

    def __init__(self, linear_accel, linear_decel, angular_accel, angular_decel):
        self.linear = SlewLimiter(linear_accel, linear_decel)
        self.angular = SlewLimiter(angular_accel, angular_decel)

    def configure(self, linear_accel, linear_decel, angular_accel, angular_decel):
        self.linear.accel, self.linear.decel = linear_accel, linear_decel
        self.angular.accel, self.angular.decel = angular_accel, angular_decel

    def reset(self):
        self.linear.reset()
        self.angular.reset()

    def step(self, linear, angular, dt):
        return self.linear.step(linear, dt), self.angular.step(angular, dt)
