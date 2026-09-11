#!/usr/bin/env python3
# ===============================================================
# OmniVLA-edge ROS2 Node
# ===============================================================
#
# - Loads the OmniVLA-edge model ONCE at node startup (not per-frame).
# - Subscribes to a camera topic to keep a rolling context of frames.
# - Subscribes to GPS (NavSatFix) + heading (Float32, degrees) for the
#   robot's *current* pose (this must be live, it changes every tick).
# - The *goal* / inference request (target GPS, target compass,
#   language prompt, goal image, which modalities to use, which
#   predicted waypoint to use, enable/disable) is entirely driven by
#   TOPICS, not parameters, so you can test it live with `ros2 topic
#   pub` or a small test publisher node while omnivla_edge_node is
#   running:
#
#     ros2 topic pub /goal_img sensor_msgs/msg/Image ...
#     ros2 topic pub /lan_prompt std_msgs/msg/String "{data: 'blue trash bin'}"
#
#   The goal and modality topics are LATCHED (see LATCHED_QOS below), so
#   `ros2 topic pub` needs --qos-durability transient_local on them or it
#   will not connect to this node at all -- QoS mismatches fail silently,
#   with no publisher-side error:
#
#     Q="--qos-durability transient_local --qos-reliability reliable"
#     ros2 topic pub $Q /goal_gps sensor_msgs/msg/NavSatFix "{latitude: 37.8739, longitude: -122.2675}"
#     ros2 topic pub $Q /goal_compass std_msgs/msg/Float32 "{data: 0.0}"
#     # (the Float32 type here was always right -- the subscription below
#     # used to declare Int32, which meant it silently never connected to a
#     # Float32 publisher. Fixed 2026-09-09. Match /erc/heading_deg's type,
#     # same reasoning as the comment on compass_topic above.)
#     ros2 topic pub $Q /use_lan_prompt std_msgs/msg/Bool "{data: true}"
#     ros2 topic pub $Q /enable_inference std_msgs/msg/Bool "{data: true}"
#
#   To navigate by GPS you must also select the pose modality, or this node
#   stays on its defaults (satellite), where the model masks the GPS goal:
#
#     ros2 topic pub $Q /use_pose_goal std_msgs/msg/Bool "{data: true}"
#     ros2 topic pub $Q /use_satellite std_msgs/msg/Bool "{data: false}"
#
#   The topic names themselves are still configurable via parameters
#   (so you can remap without touching code), only their *values* are
#   no longer ROS2 parameters.
#
# - Runs inference on a timer (tick_rate Hz) and publishes the result
#   as a geometry_msgs/Twist on /cmd_vel.
#
# Place this file in a ROS2 python package (alongside model_omnivla_edge.py
# and utils_policy.py, which must be importable), add it as an entry
# point / node, and launch it.
# ===============================================================

import math
import threading
from collections import deque

import numpy as np
import torch
import utm
import clip
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Float32, String, Bool, Int32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from erc_inference.utils_policy import (
    transform_images_map,
    load_model,
    transform_images_PIL_mask,
)

IMG_SIZE = (96, 96)
IMG_SIZE_CLIP = (224, 224)
# Scale between the model's goal/waypoint units and meters. 0.25 is what
# LogoNav was trained with on FrodoBots data (hardcoded in MBRA's
# vint_hf_dataset.py:662), both for the goal it reads and for the waypoints it
# predicts. This node used 0.1, which fed goals 2.5x too large and shrank the
# predicted waypoints 2.5x. Independent check from mission9sept: the index-4
# waypoint (+1.5 s) came out as dx ~= 0.58 m at 0.1, i.e. 1.46 m at 0.25 --
# 0.97 m/s, the Mini+'s typical speed in that dataset -- against 0.39 m/s.
METRIC_WAYPOINT_SPACING = 0.25
THRES_DIST = 30.0
# A goal that moves further than this between two messages is a new target (a
# new leg, or a replan), not the carrot's normal advance of a few cm per tick.
GOAL_RESET_DISTANCE_M = 3.0

# Latched state, not events -- the goal (a carrot checkpoint_controller_node
# moves along the route) and the modality flags (sent once per leg). With
# volatile QoS this node receives nothing if it starts (or restarts) mid-leg and silently
# falls back to the defaults below, which select modality 0 (satellite): the
# model then masks the GPS goal out entirely and the rover drives blind.
# TRANSIENT_LOCAL delivers the current value to a late joiner. Both ends must
# agree -- a volatile publisher never connects to a transient-local subscriber
# -- so checkpoint_controller_node.py and prueba.py declare the same profile.
LATCHED_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


# Seconds between consecutive waypoints inside the model's predicted chunk.
# LogoNav trains with action_spacing=3 on 10 Hz data, so the 8 waypoints of the
# chunk sit at +0.3 s ... +2.4 s. This is NOT the controller tick period: the
# old control law divided by 1/tick_rate (0.333 s) as if the selected waypoint
# had to be reached within one tick, which over-commanded both channels by
# (waypoint_select + 1) * 0.3 / 0.333 -- 4.5x at waypoint_select=4.
WAYPOINT_DT = 0.3


def clip_angle(angle: float) -> float:
    """Wrap an angle (rad) to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def ground_distance_m(lat1, lon1, lat2, lon2):
    """Equirectangular distance -- ample for telling a carrot step from a new leg."""
    k = math.radians(1.0) * 6378137.0
    east = (lon2 - lon1) * k * math.cos(math.radians(0.5 * (lat1 + lat2)))
    return math.hypot(east, (lat2 - lat1) * k)


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


CONTROLLER_TYPES = ("polar", "pid")


def polar_control(dx, dy, hx, hy, k_rho, k_alpha, k_beta, max_linear, max_angular,
                  backward_allowed=False, use_constant_vel=False, constant_vel=0.1):
    """Siegwart's polar-coordinate controller, applied to one predicted waypoint.

    Siegwart & Nourbakhsh, "Introduction to Autonomous Mobile Robots", 3.6.2.
    Everything is in the robot frame the model predicts in: the robot sits at
    the origin with heading 0, and the goal pose is the selected waypoint --
    position (dx, dy) and heading atan2(hy, hx). LogoNav is trained with
    learn_angle=True, so the model predicts where the rover should be
    pointing at that waypoint as well as where it should be; this is the
    controller that uses that half of the prediction.

        rho   = distance to the waypoint
        alpha = bearing of the waypoint relative to the robot's heading
        beta  = heading the model wants at the waypoint, minus that bearing
        v     = k_rho * rho
        w     = k_alpha * alpha + k_beta * beta

    Deliberate differences from a straight port of the usual implementation:
    - w is clipped symmetrically to +-max_angular. Clipping only the upper
      side leaves every right turn unbounded.
    - v is clipped to max_linear; k_rho * rho has no bound of its own.
    - beta is wrapped to [-pi, pi] like alpha.
    - Driving backward redefines the robot's forward axis (alpha += pi,
      v < 0), which is Siegwart's construction. Choosing the sign of w by
      comparing the goal bearing with the goal heading never actually
      reverses the rover.
    - No "goal reached, stop" latch. The waypoint is re-predicted every tick
      ~1.5 s ahead of the rover, so rho does not approach zero in normal
      driving, and arrival is decided by checkpoint_controller_node against
      GPS; a latch here would stop the rover and fight that decision.

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

    speed = constant_vel if use_constant_vel else k_rho * rho
    v_min = -max_linear if backward_allowed else 0.0
    v = float(np.clip(direction * speed, v_min, max_linear))
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


class GoalTurn:
    """Turn toward the goal when it is behind the rover, before the model drives.

    The model cannot ask for this. Its selected waypoint always sits in the
    forward half-plane -- on mission_10sept its bearing never went beyond
    +-18 deg -- so whenever a leg starts with the goal behind, any controller
    that follows the model drives away from it. That is how the first attempt
    at leg 3 of mission_10sept ended (goal ~175 deg behind, rover drove on from
    15 m to 24.5 m, aborted by hand), and leg 3 of mission9sept before it: cp3
    is back at cp1, so the return leg always opens with a U-turn.

    Driven by localization, not by the model: the bearing comes from GPS and the
    magnetometer heading, validated on mission_10sept at -5.8 deg bias against
    GPS course over all headings. Hysteresis keeps it from chattering: engage
    past enter_deg, release under exit_deg. The rover keeps rotating for ~1 s
    after the command drops (cmd->gyro lag measured 0.85-1.4 s), which is why
    exit_deg is well short of zero rather than a sign the turn is incomplete.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.direction = 0.0

    def update(self, bearing, distance, enabled, enter_deg, exit_deg, min_distance_m):
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
            # Latched when engaging: near 180 deg the sign of the bearing flips
            # with every bit of heading noise, and re-reading it each tick would
            # reverse the turn halfway through.
            self.direction = 1.0 if bearing >= 0.0 else -1.0
        return self.active


class OmniVLAEdgeNode(Node):
    def __init__(self):
        super().__init__("omnivla_edge_node")
        self.bridge = CvBridge()
        self.lock = threading.RLock()

        # ---------------------------------------------------------
        # Static / startup-only parameters (model + topics)
        # ---------------------------------------------------------
        self.declare_parameter("model_checkpoint_path", "./omnivla-edge/omnivla-edge.pth")
        self.declare_parameter("context_size", 5)
        self.declare_parameter("obs_encoder", "efficientnet-b0")
        self.declare_parameter("encoding_size", 256)
        self.declare_parameter("obs_encoding_size", 1024)
        self.declare_parameter("goal_encoding_size", 1024)
        self.declare_parameter("late_fusion", False)
        self.declare_parameter("mha_num_attention_heads", 4)
        self.declare_parameter("mha_num_attention_layers", 4)
        self.declare_parameter("mha_ff_dim_factor", 4)
        self.declare_parameter("clip_type", "ViT-B/32")
        self.declare_parameter("len_traj_pred", 8)
        self.declare_parameter("learn_angle", True)

        self.declare_parameter("image_topic", "/erc/front_camera")
        # The EKF-filtered fix (navsat_transform, 10 Hz, 0.19 m p50 from raw GPS
        # on mission_10sept) rather than the raw SDK fix, which refreshes every
        # ~1.3 s in 0.4 m steps. Against a carrot 1.5 m ahead a 0.4 m step
        # swings the goal bearing by ~15 deg. Needs erc_localization's
        # localization_global.launch.py running.
        self.declare_parameter("gps_topic", "/erc/gps/filtered")
        # Unified heading topic from erc_localization's heading_node.py, which
        # owns the magnetometer-vs-SDK-compass choice (config/heading.yaml).
        # Float32 rather than the Int32 /erc/orientation used to be: the
        # magnetometer path resolves finer than a whole degree, and the value
        # keeps the SDK's own convention (0 = North, clockwise-positive), which
        # is what the cur_compass arithmetic below was written against and what
        # OmniVLA was trained on. Do not "fix" that to ENU here.
        self.declare_parameter("compass_topic", "/erc/heading_deg")
        self.declare_parameter("cmd_vel_topic", "/omnivla/cmd_vel")
        self.declare_parameter("tick_rate", 3.0)

        # Velocity limits stay as parameters: they're safety bounds for
        # the robot, not part of the "goal" you're testing/iterating on.
        self.declare_parameter("max_linear_vel", 0.3)
        self.declare_parameter("max_angular_vel", 0.3)

        # Which law turns the selected waypoint into /cmd_vel: "polar" or
        # "pid". Normally set from config/controller.yaml; can be switched
        # live, and _on_set_parameters rejects anything else.
        self.declare_parameter("controller_type", "polar")

        # --- Heading PID (see HeadingPID) -----------------------------------
        # kp 0.4 puts the angular command at the 0.3 rad/s limit for a bearing
        # error of ~43 deg, against 5.73 deg under the old arctan/DT law: the
        # output stops being effectively three-valued and gets a real
        # proportional band.
        #
        # These were picked for ROBUSTNESS TO THE LOOP DELAY, which is only
        # known to within a factor of two -- mission9sept measured cmd_delay at
        # 0.246 s but the best cmd->gyro correlation sat at a lag of 1.15 s.
        # Simulated against a delay+first-order plant at both measured
        # actuation ratios (0.98 and 0.50), worst-case overshoot over 30/90/180
        # deg steps:
        #
        #   kp    ki    kd  | delay 0.6s | 0.9s        | 1.2s
        #   0.5   0.15  0.08|  9.3 (24x) | 13.0 (35x)  | 18.1 (32x)
        #   0.6   0.03  0.10|  1.6 ( 1x) |  5.4 ( 3x)  | 10.9 ( 5x)
        #   0.6   0.00  0.10|  0.0 ( 0x) |  4.6 (24x)  | 10.1 (39x)
        #   0.4   0.03  0.10|  2.4 ( 4x) |  2.5 ( 4x)  |  4.7 ( 4x)  <- chosen
        #
        # (parenthesis = zero crossings, i.e. weaving). The gains that win at
        # the nominal delay are close to unstable at the pessimistic one; 0.4
        # is barely slower to settle (6.0 s vs 5.3 s on a 90 deg step) and
        # holds its behaviour across the whole band.
        #
        # ki is small on purpose: the rover does not always execute what it is
        # told, and a large integral winds up against a deficit it cannot fix.
        # Set ki to 0 if the rover still weaves on real terrain -- that costs
        # steady-state bias rejection but cannot ring.
        self.declare_parameter("pid.kp", 0.4)
        self.declare_parameter("pid.ki", 0.03)
        self.declare_parameter("pid.kd", 0.10)
        # Cap on the integral STATE (rad*s). At ki=0.03 this bounds the
        # integral's contribution to ~0.02 rad/s.
        self.declare_parameter("pid.integral_limit", 0.67)
        self.declare_parameter("pid.derivative_alpha", 0.4)

        # --- Linear speed shaping -------------------------------------------
        # Deliberately gentle. Measured on mission9sept: the achieved/commanded
        # yaw-rate ratio was 0.98 on gentle turns at 0.3 m/s but only 0.29 when
        # the old limiter had slowed the rover mid-turn, and 0.50 at cruise
        # within that same leg -- i.e. slowing down made the rover turn WORSE,
        # then it stayed saturated, which slowed it further. Speed is therefore
        # only reduced once the goal is far enough off-axis that driving
        # forward stops closing the distance at all (past 90 deg it opens it).
        self.declare_parameter("pid.turn_slowdown_start_deg", 60.0)
        self.declare_parameter("pid.turn_slowdown_end_deg", 120.0)
        self.declare_parameter("pid.turn_speed_floor", 0.4)

        # --- Polar controller (see polar_control) ---------------------------
        # k_alpha 1.5 against pid.kp 0.4 is the point of this controller. On
        # mission_10sept the PID never commanded more than 0.153 rad/s -- the
        # model's waypoint bearing stays small -- and turned on a measured
        # 4.9 m median radius. Replayed on those same waypoints, the commanded
        # radius drops from 6.3 m to 2.7 m here. k_beta is 0 rather than
        # Siegwart's negative value: -0.6 opened that to 3.2 m (see
        # polar_gain_warnings for why 0 is valid).
        self.declare_parameter("polar.k_rho", 0.5)
        self.declare_parameter("polar.k_alpha", 1.5)
        self.declare_parameter("polar.k_beta", 0.0)
        self.declare_parameter("polar.backward_allowed", False)
        self.declare_parameter("polar.use_constant_vel", False)
        self.declare_parameter("polar.constant_vel", 0.1)

        # --- Goal behind the rover (see GoalTurn) ---------------------------
        # Overrides whichever controller is active: past enter_deg of bearing
        # to the goal, turn at angular_vel (linear_vel forward; 0 = in place,
        # which ran at 1.08x of command in wuhan4_angular) until under
        # exit_deg, then hand back.
        self.declare_parameter("goal_turn.enabled", True)
        self.declare_parameter("goal_turn.enter_deg", 90.0)
        self.declare_parameter("goal_turn.exit_deg", 30.0)
        self.declare_parameter("goal_turn.angular_vel", 0.3)
        self.declare_parameter("goal_turn.linear_vel", 0.0)
        self.declare_parameter("goal_turn.min_distance_m", 1.0)

        # ---------------------------------------------------------
        # Topic names for the goal / inference request. The VALUES
        # come from messages on these topics (see subscriptions
        # below), not from parameters, so you can drive them live
        # with `ros2 topic pub` for testing.
        # ---------------------------------------------------------
        self.declare_parameter("debug_topic", "/omnivla_debug")
        self.declare_parameter("goal_image_topic", "/goal_img")
        self.declare_parameter("goal_gps_topic", "/goal_gps")
        self.declare_parameter("goal_compass_topic", "/goal_compass")
        self.declare_parameter("lan_prompt_topic", "/lan_prompt")
        self.declare_parameter("use_pose_goal_topic", "/use_pose_goal")
        self.declare_parameter("use_satellite_topic", "/use_satellite")
        self.declare_parameter("use_image_goal_topic", "/use_image_goal")
        self.declare_parameter("use_lan_prompt_topic", "/use_lan_prompt")
        self.declare_parameter("waypoint_select_topic", "/waypoint_select")
        self.declare_parameter("enable_inference_topic", "/enable_inference")

        # ---------------------------------------------------------
        # Load model ONCE
        # ---------------------------------------------------------
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Loading OmniVLA-edge on {self.device} ...")

        model_params = {
            "model_type": "omnivla-edge",
            "len_traj_pred": self.get_parameter("len_traj_pred").value,
            "learn_angle": self.get_parameter("learn_angle").value,
            "context_size": self.get_parameter("context_size").value,
            "obs_encoder": self.get_parameter("obs_encoder").value,
            "encoding_size": self.get_parameter("encoding_size").value,
            "obs_encoding_size": self.get_parameter("obs_encoding_size").value,
            "goal_encoding_size": self.get_parameter("goal_encoding_size").value,
            "late_fusion": self.get_parameter("late_fusion").value,
            "mha_num_attention_heads": self.get_parameter("mha_num_attention_heads").value,
            "mha_num_attention_layers": self.get_parameter("mha_num_attention_layers").value,
            "mha_ff_dim_factor": self.get_parameter("mha_ff_dim_factor").value,
            "clip_type": self.get_parameter("clip_type").value,
        }
        ckpt_path = self.get_parameter("model_checkpoint_path").value
        self.model, self.text_encoder, self.preprocess = load_model(ckpt_path, model_params, self.device)
        self.model = self.model.to(self.device).eval()
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.context_size = model_params["context_size"]
        self.get_logger().info("Model loaded.")

        # No mask (fisheye-specific masking disabled, matches sample script default)
        self.mask_96 = np.ones((96, 96, 3), dtype=np.float32)
        self.mask_224 = np.ones((224, 224, 3), dtype=np.float32)

        # ---------------------------------------------------------
        # Mutable state, protected by self.lock
        # ---------------------------------------------------------
        self.context_queue = deque(maxlen=self.context_size + 1)
        self.latest_frame_full = None  # for the 224px "cur_large_img"
        self.current_lat = None
        self.current_lon = None
        self.current_compass_deg = None

        # Goal / inference-request state, all driven by topics. Sane
        # defaults so the node doesn't crash before the first message
        # arrives on each topic; enable_inference defaults True so it
        # "just runs" as soon as sensing + a goal image are available,
        # which is convenient for testing.
        self.goal_image_pil = PILImage.new("RGB", IMG_SIZE, color=(0, 0, 0))
        # None, not 0.0: (0.0, 0.0) is a real coordinate in the Gulf of Guinea,
        # so a 0.0 default is indistinguishable from a goal that was actually
        # received and would aim the rover at a bearing ~12,700 km away (clamped
        # to THRES_DIST, so it looks like a plausible 30 m goal rather than
        # failing loudly). None makes "no goal yet" detectable in timer_callback.
        self.goal_lat = None
        self.goal_lon = None
        self.goal_compass_deg = 0.0
        self.lan_inst_prompt = ""
        self.use_pose_goal = False
        self.use_satellite = True
        self.use_image_goal = False
        self.use_lan_prompt = False
        self.waypoint_select = 4
        self.enable_inference = True

        self.heading_pid = HeadingPID(
            kp=self.get_parameter("pid.kp").value,
            ki=self.get_parameter("pid.ki").value,
            kd=self.get_parameter("pid.kd").value,
            out_limit=self.get_parameter("max_angular_vel").value,
            integral_limit=self.get_parameter("pid.integral_limit").value,
            derivative_alpha=self.get_parameter("pid.derivative_alpha").value,
        )

        controller_type = self.get_parameter("controller_type").value
        if controller_type not in CONTROLLER_TYPES:
            raise ValueError(f"controller_type must be one of {CONTROLLER_TYPES}, "
                             f"got {controller_type!r} (check config/controller.yaml)")
        self._last_controller_type = None
        for warning in polar_gain_warnings(self.get_parameter("polar.k_rho").value,
                                           self.get_parameter("polar.k_alpha").value,
                                           self.get_parameter("polar.k_beta").value):
            self.get_logger().warn(f"polar gains: {warning}")
        self.add_on_set_parameters_callback(self._on_set_parameters)
        # Real elapsed time between ticks: the forward pass took 168 ms in
        # profiling and the timer is not guaranteed to fire on schedule, so
        # feeding the PID a nominal 1/tick_rate would misstate both the
        # integral and the derivative.
        self._last_tick_time = None
        self.goal_turn = GoalTurn()

        # ---------------------------------------------------------
        # Pub / Sub
        # ---------------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.debug_pub = self.create_publisher(String, self.get_parameter("debug_topic").value, 10)

        # Live robot state
        self.create_subscription(Image, self.get_parameter("image_topic").value, self.image_callback, 10)
        self.create_subscription(NavSatFix, self.get_parameter("gps_topic").value, self.gps_callback, 10)
        self.create_subscription(Float32, self.get_parameter("compass_topic").value, self.compass_callback, 10)

        # Goal / inference request (test these live with `ros2 topic pub`)
        self.create_subscription(Image, self.get_parameter("goal_image_topic").value, self.goal_image_callback, 10)
        self.create_subscription(NavSatFix, self.get_parameter("goal_gps_topic").value, self.goal_gps_callback, LATCHED_QOS)
        self.create_subscription(Float32, self.get_parameter("goal_compass_topic").value, self.goal_compass_callback, LATCHED_QOS)
        self.create_subscription(String, self.get_parameter("lan_prompt_topic").value, self.lan_prompt_callback, 10)
        self.create_subscription(Bool, self.get_parameter("use_pose_goal_topic").value, self.use_pose_goal_callback, LATCHED_QOS)
        self.create_subscription(Bool, self.get_parameter("use_satellite_topic").value, self.use_satellite_callback, LATCHED_QOS)
        self.create_subscription(Bool, self.get_parameter("use_image_goal_topic").value, self.use_image_goal_callback, LATCHED_QOS)
        self.create_subscription(Bool, self.get_parameter("use_lan_prompt_topic").value, self.use_lan_prompt_callback, LATCHED_QOS)
        self.create_subscription(Int32, self.get_parameter("waypoint_select_topic").value, self.waypoint_select_callback, 10)
        self.create_subscription(Bool, self.get_parameter("enable_inference_topic").value, self.enable_inference_callback, LATCHED_QOS)

        tick_rate = self.get_parameter("tick_rate").value
        self.timer = self.create_timer(1.0 / tick_rate, self.timer_callback)

        self.get_logger().info(
            "OmniVLA-edge node ready. Publish to the goal topics (e.g. "
            f"{self.get_parameter('goal_image_topic').value}) at any time to update the inference request."
        )

    # ---------------------------------------------------------
    # Sensor callbacks (live robot state)
    # ---------------------------------------------------------
    def image_callback(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        pil_img = PILImage.fromarray(cv_img)
        with self.lock:
            self.latest_frame_full = pil_img.resize(IMG_SIZE_CLIP)
            self.context_queue.append(pil_img.resize(IMG_SIZE))

    def gps_callback(self, msg: NavSatFix):
        with self.lock:
            self.current_lat = msg.latitude
            self.current_lon = msg.longitude

    def compass_callback(self, msg: Float32):
        with self.lock:
            self.current_compass_deg = msg.data

    # ---------------------------------------------------------
    # Goal / inference-request callbacks (this is what you test with
    # `ros2 topic pub`)
    # ---------------------------------------------------------
    def goal_image_callback(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        pil_img = PILImage.fromarray(cv_img).resize(IMG_SIZE)
        with self.lock:
            self.goal_image_pil = pil_img
        self.get_logger().info("Goal image updated.")

    def goal_gps_callback(self, msg: NavSatFix):
        with self.lock:
            # checkpoint_controller_node streams a carrot that advances a few
            # cm per tick; only a new leg or a replan moves it by meters. Reset
            # on those jumps only: resetting on every carrot step would leave
            # the PID with no integral or derivative at all, and a new target
            # needs the goal-turn direction latched afresh.
            jumped = (self.goal_lat is None or ground_distance_m(
                self.goal_lat, self.goal_lon, msg.latitude, msg.longitude) > GOAL_RESET_DISTANCE_M)
            self.goal_lat = msg.latitude
            self.goal_lon = msg.longitude
            if jumped:
                self.heading_pid.reset()
                self.goal_turn.reset()
        if jumped:
            self.get_logger().info(f"New goal: lat={msg.latitude}, lon={msg.longitude}")

    def goal_compass_callback(self, msg: Float32):
        with self.lock:
            self.goal_compass_deg = msg.data

    def lan_prompt_callback(self, msg: String):
        with self.lock:
            self.lan_inst_prompt = msg.data
        self.get_logger().info(f"Language prompt updated: '{msg.data}'")

    def use_pose_goal_callback(self, msg: Bool):
        with self.lock:
            self.use_pose_goal = msg.data

    def use_satellite_callback(self, msg: Bool):
        with self.lock:
            self.use_satellite = msg.data

    def use_image_goal_callback(self, msg: Bool):
        with self.lock:
            self.use_image_goal = msg.data

    def use_lan_prompt_callback(self, msg: Bool):
        with self.lock:
            self.use_lan_prompt = msg.data

    def waypoint_select_callback(self, msg: Int32):
        with self.lock:
            self.waypoint_select = msg.data

    def enable_inference_callback(self, msg: Bool):
        with self.lock:
            was_enabled = self.enable_inference
            self.enable_inference = msg.data
            # The rover is stopped while disabled (checkpoint verification, and
            # the gaps between legs lasted 1.4 s in mission9sept). Resuming with
            # the integral from before the stop would kick on the first tick.
            if was_enabled != msg.data:
                self.heading_pid.reset()
                self.goal_turn.reset()
                self._last_tick_time = None
        self.get_logger().info(f"enable_inference set to {msg.data}")

    # ---------------------------------------------------------
    # Helpers (ported from run_omnivla_edge.py)
    # ---------------------------------------------------------
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    @staticmethod
    def compute_modality_id(pose_goal, satellite, image_goal, lan_prompt):
        if pose_goal and satellite and image_goal and not lan_prompt:
            return 3
        elif not pose_goal and satellite and not image_goal and not lan_prompt:
            return 0
        elif pose_goal and not satellite and not image_goal and not lan_prompt:
            return 4
        elif pose_goal and satellite and not image_goal and not lan_prompt:
            return 1
        elif not pose_goal and satellite and image_goal and not lan_prompt:
            return 2
        elif pose_goal and not satellite and image_goal and not lan_prompt:
            return 5
        elif not pose_goal and not satellite and image_goal and not lan_prompt:
            return 6
        elif not pose_goal and not satellite and not image_goal and lan_prompt:
            return 7
        elif pose_goal and not satellite and not image_goal and lan_prompt:
            return 8
        elif not pose_goal and not satellite and image_goal and lan_prompt:
            return 9
        # Fallback: no modality selected -> treat as satellite-only
        return 0

    # ---------------------------------------------------------
    # Main inference tick
    # ---------------------------------------------------------
    def timer_callback(self):
        with self.lock:
            # A pose goal is only required by the modalities that actually read
            # the goal_pose token (use_pose_goal); the satellite/image/language
            # modalities have it masked out inside the model, so gating them on
            # a GPS goal would stall them for no reason.
            have_pose_goal = self.goal_lat is not None and self.goal_lon is not None
            ready = (
                self.enable_inference
                and len(self.context_queue) == self.context_size + 1
                and self.latest_frame_full is not None
                and self.current_lon is not None
                and self.current_compass_deg is not None
                and (have_pose_goal or not self.use_pose_goal)
            )
            if not ready:
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info("Inference not ready: waiting for context, latest frame, and current pose.")
                #info who is not ready
                if self.enable_inference:
                    self.get_logger().info(f"enable_inference={self.enable_inference}, context_queue={len(self.context_queue)}/{self.context_size + 1}, latest_frame_full={self.latest_frame_full is not None}, current_lat={self.current_lat is not None}, current_lon={self.current_lon is not None}, current_compass_deg={self.current_compass_deg is not None}, goal_gps={have_pose_goal} (required={self.use_pose_goal})")

                return


            # Snapshot everything we need under the lock, then release it
            # before running the (potentially slow) forward pass.
            context_queue = list(self.context_queue)
            cur_large_pil = self.latest_frame_full
            current_lat = self.current_lat
            current_lon = self.current_lon
            current_compass_deg = self.current_compass_deg

            # Reachable with no goal only when use_pose_goal is False, i.e. the
            # model masks this token anyway. Fall back to the current position so
            # the goal vector is (0, 0) instead of feeding NaN/None into UTM.
            goal_lat = self.goal_lat if self.goal_lat is not None else self.current_lat
            goal_lon = self.goal_lon if self.goal_lon is not None else self.current_lon
            goal_compass_deg = self.goal_compass_deg
            lan_inst_prompt = self.lan_inst_prompt
            goal_image_pil = self.goal_image_pil

            use_pose_goal = self.use_pose_goal
            use_satellite = self.use_satellite
            use_image_goal = self.use_image_goal
            use_lan_prompt = self.use_lan_prompt

            waypoint_select = self.waypoint_select
            max_linear = self.get_parameter("max_linear_vel").value
            max_angular = self.get_parameter("max_angular_vel").value

        try:
            linear_vel, angular_vel = self.run_inference(
                context_queue,
                cur_large_pil,
                current_lat,
                current_lon,
                current_compass_deg,
                goal_lat,
                goal_lon,
                goal_compass_deg,
                lan_inst_prompt,
                goal_image_pil,
                use_pose_goal,
                use_satellite,
                use_image_goal,
                use_lan_prompt,
                waypoint_select,
                max_linear,
                max_angular,
            )
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            self.publish_cmd(0.0, 0.0)
            return

        self.publish_cmd(linear_vel, angular_vel)

    def run_inference(
        self,
        context_queue,
        cur_large_pil,
        current_lat,
        current_lon,
        current_compass_deg,
        goal_lat,
        goal_lon,
        goal_compass_deg,
        lan_inst_prompt,
        goal_image_pil,
        use_pose_goal,
        use_satellite,
        use_image_goal,
        use_lan_prompt,
        waypoint_select,
        max_linear,
        max_angular,
    ):
        device = self.device

        # --- Current pose ---
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass_deg) / 180.0 * math.pi

        # --- Goal pose (relative, local frame) ---
        goal_utm = utm.from_latlon(goal_lat, goal_lon)
        goal_compass = -float(goal_compass_deg) / 180.0 * math.pi

        delta_x, delta_y = self.calculate_relative_position(cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1])
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x ** 2 + relative_y ** 2)
        # Where the goal really is, from localization alone (forward = +rel_y,
        # left = -rel_x: the same axes goal_pose_torch hands the model below).
        # GoalTurn acts on this, since the model's waypoint cannot point behind.
        goal_bearing = math.atan2(-relative_x, relative_y)
        if radius > THRES_DIST:
            relative_x *= THRES_DIST / radius
            relative_y *= THRES_DIST / radius

        goal_pose_torch = torch.from_numpy(np.array([
            relative_y / METRIC_WAYPOINT_SPACING,
            -relative_x / METRIC_WAYPOINT_SPACING,
            np.cos(goal_compass - cur_compass),
            np.sin(goal_compass - cur_compass),
        ])).unsqueeze(0).float().to(device)

        # --- Observation context ---
        obs_images = transform_images_PIL_mask(context_queue, self.mask_96)
        obs_images = torch.split(obs_images.to(device), 3, dim=1)
        obs_image_cur = obs_images[-1].to(device)
        obs_images = torch.cat(obs_images, dim=1).to(device)

        cur_large_img = transform_images_PIL_mask(cur_large_pil, self.mask_224).to(device)

        # --- Satellite imagery (dummy black placeholder; wire up a real
        # satellite/aerial tile subscriber here if use_satellite is used
        # in your deployment) ---
        satellite_cur = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
        satellite_goal = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
        current_map_image = transform_images_map(satellite_cur)
        goal_map_image = transform_images_map(satellite_goal)
        map_images = torch.cat((current_map_image.to(device), goal_map_image.to(device), obs_image_cur), axis=1)

        # --- Language instruction ---
        lan_inst = lan_inst_prompt if (use_lan_prompt and lan_inst_prompt) else "xxxx"
        obj_inst_lan = clip.tokenize(lan_inst, truncate=True).to(device)

        # --- Egocentric goal image ---
        goal_image = transform_images_PIL_mask(goal_image_pil, self.mask_96).to(device)

        modality_id = self.compute_modality_id(use_pose_goal, use_satellite, use_image_goal, use_lan_prompt)
        modality_id_select = torch.tensor([modality_id]).to(device)

        bimg = goal_image.size(0)
        with torch.no_grad():
            feat_text_lan = self.text_encoder.encode_text(obj_inst_lan)
            predicted_actions, distances, mask_number = self.model(
                obs_images.repeat(bimg, 1, 1, 1),
                goal_pose_torch.repeat(bimg, 1),
                map_images.repeat(bimg, 1, 1, 1),
                goal_image,
                modality_id_select.repeat(bimg),
                feat_text_lan.repeat(bimg, 1),
                cur_large_img.repeat(bimg, 1, 1, 1),
            )

        waypoints = predicted_actions.float().cpu().numpy()
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= METRIC_WAYPOINT_SPACING
        dx, dy, hx, hy = chosen_waypoint

        self.get_logger().info(
            f"[modality={modality_id}] raw_waypoint(dx={dx:.3f}, dy={dy:.3f}, hx={hx:.3f}, hy={hy:.3f}) "
            f"waypoint_select={waypoint_select} all_waypoints_shape={waypoints.shape}"
        )

        # --- Goal behind the rover? (see GoalTurn) ---
        # Decided before the controller runs, so the PID is neither stepped
        # during the turn nor handed its pre-turn history on the tick it ends.
        gp = self.get_parameter
        was_turning = self.goal_turn.active
        if use_pose_goal:
            turning = self.goal_turn.update(
                goal_bearing, float(radius),
                enabled=gp("goal_turn.enabled").value,
                enter_deg=gp("goal_turn.enter_deg").value,
                exit_deg=gp("goal_turn.exit_deg").value,
                min_distance_m=gp("goal_turn.min_distance_m").value)
        else:
            self.goal_turn.reset()
            turning = False
        if was_turning and not turning:
            self.heading_pid.reset()
            self._last_tick_time = None

        # --- Controller -> (linear, angular) ---
        controller_type = self.get_parameter("controller_type").value
        if controller_type != self._last_controller_type:
            # Switching laws mid-drive: the PID's integral and derivative
            # history describe the other law's commands, and its dt clock has
            # been idle while polar ran.
            self.heading_pid.reset()
            self._last_tick_time = None
            self.get_logger().info(f"Motion controller: {controller_type}")
            self._last_controller_type = controller_type

        if turning:
            mode = "goal-turn"
            linear_cmd = float(np.clip(gp("goal_turn.linear_vel").value, 0.0, max_linear))
            angular_cmd = self.goal_turn.direction * min(abs(gp("goal_turn.angular_vel").value), max_angular)
            detail = "controller bypassed"
        elif controller_type == "polar":
            mode = controller_type
            linear_cmd, angular_cmd, detail = self._polar_command(
                dx, dy, hx, hy, max_linear, max_angular)
        elif controller_type == "pid":
            mode = controller_type
            linear_cmd, angular_cmd, detail = self._pid_command(
                dx, dy, hx, hy, waypoint_select, max_linear, max_angular)
        else:
            # __init__ refuses to start with this and _on_set_parameters
            # rejects it live, so reaching here means both were bypassed.
            # Raising makes timer_callback stop the rover rather than guess.
            raise ValueError(f"unknown controller_type {controller_type!r}")

        debug_msg = (
            f"[{mode}] goal_bearing={math.degrees(goal_bearing):+.1f}deg goal_dist={radius:.2f}m | "
            f"{detail} | linear_cmd={linear_cmd:.4f} angular_cmd={angular_cmd:.4f} | "
            f"modality={modality_id} lan_prompt='{lan_inst}'"
        )
        self.get_logger().info(debug_msg)
        self.debug_pub.publish(String(data=debug_msg))

        return float(linear_cmd), float(angular_cmd)

    def _on_set_parameters(self, params):
        """Reject an unknown controller_type; warn when polar gains go unstable."""
        gains = {name: self.get_parameter(name).value
                 for name in ("polar.k_rho", "polar.k_alpha", "polar.k_beta")}
        touched_gains = False
        for param in params:
            if param.name == "controller_type" and param.value not in CONTROLLER_TYPES:
                return SetParametersResult(
                    successful=False,
                    reason=f"controller_type must be one of {CONTROLLER_TYPES}, got {param.value!r}")
            if param.name in gains:
                gains[param.name] = param.value
                touched_gains = True
        if touched_gains:
            for warning in polar_gain_warnings(gains["polar.k_rho"], gains["polar.k_alpha"],
                                               gains["polar.k_beta"]):
                self.get_logger().warn(f"polar gains: {warning}")
        return SetParametersResult(successful=True)

    def _polar_command(self, dx, dy, hx, hy, max_linear, max_angular):
        gp = self.get_parameter
        v, w, rho, alpha, beta = polar_control(
            dx, dy, hx, hy,
            k_rho=gp("polar.k_rho").value,
            k_alpha=gp("polar.k_alpha").value,
            k_beta=gp("polar.k_beta").value,
            max_linear=max_linear,
            max_angular=max_angular,
            backward_allowed=gp("polar.backward_allowed").value,
            use_constant_vel=gp("polar.use_constant_vel").value,
            constant_vel=gp("polar.constant_vel").value,
        )
        detail = (f"rho={rho:.3f}m alpha={math.degrees(alpha):+.1f}deg "
                  f"beta={math.degrees(beta):+.1f}deg")
        return v, w, detail

    def _pid_command(self, dx, dy, hx, hy, waypoint_select, max_linear, max_angular):
        EPS = 1e-8
        maxv = max_linear

        # Refresh gains from parameters every tick so they can be tuned live
        # with `ros2 param set` while the rover drives, which is the only
        # practical way to tune this on real terrain. Cheap next to the
        # forward pass that just ran.
        self.heading_pid.kp = self.get_parameter("pid.kp").value
        self.heading_pid.ki = self.get_parameter("pid.ki").value
        self.heading_pid.kd = self.get_parameter("pid.kd").value
        self.heading_pid.integral_limit = self.get_parameter("pid.integral_limit").value
        self.heading_pid.derivative_alpha = self.get_parameter("pid.derivative_alpha").value
        self.heading_pid.out_limit = max_angular

        # Measured tick period, not the nominal one (see _last_tick_time).
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_tick_time is None:
            dt = 1.0 / self.get_parameter("tick_rate").value
        else:
            dt = now - self._last_tick_time
        self._last_tick_time = now

        # Bearing to the selected waypoint, and how far away it is. atan2 (not
        # atan(dy/dx)) so a waypoint behind the rover gives an error near +-pi
        # instead of folding into the forward half-plane -- the old law could
        # not express "the goal is behind you" at all, which is exactly the
        # case it failed on in leg 3 of mission9sept.
        if abs(dx) < EPS and abs(dy) < EPS:
            # Degenerate chunk: no displacement predicted. Fall back to the
            # predicted heading versor and do not drive forward.
            heading_error = clip_angle(np.arctan2(hy, hx))
            distance = 0.0
        else:
            heading_error = clip_angle(np.arctan2(dy, dx))
            distance = float(np.hypot(dx, dy))

        angular_vel_limit, p_term, i_term, d_term = self.heading_pid.step(heading_error, dt)

        # Linear: the chunk says where the rover should be at
        # +(waypoint_select + 1) * WAYPOINT_DT seconds, so the speed it implies
        # is distance / that horizon.
        horizon_s = (waypoint_select + 1) * WAYPOINT_DT
        linear_vel_value = distance / horizon_s

        # Gentle off-axis slowdown -- full speed while the goal is roughly
        # ahead, tapering to turn_speed_floor once it is far enough off-axis
        # that driving forward no longer closes the distance. Deliberately NOT
        # proportional to the angular command: on this rover, slowing mid-turn
        # measured worse yaw tracking, not better (see turn_slowdown_start_deg).
        slow_start = math.radians(self.get_parameter("pid.turn_slowdown_start_deg").value)
        slow_end = math.radians(self.get_parameter("pid.turn_slowdown_end_deg").value)
        floor = self.get_parameter("pid.turn_speed_floor").value
        abs_err = abs(heading_error)
        if abs_err <= slow_start:
            speed_scale = 1.0
        elif abs_err >= slow_end:
            speed_scale = floor
        else:
            t = (abs_err - slow_start) / max(slow_end - slow_start, EPS)
            speed_scale = 1.0 + t * (floor - 1.0)

        linear_vel_limit = float(np.clip(linear_vel_value * speed_scale, 0.0, maxv))

        detail = (
            f"heading_err={math.degrees(heading_error):+.1f}deg dist={distance:.3f}m "
            f"dt={dt:.3f}s | P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f} | "
            f"linear_raw={linear_vel_value:.4f} scale={speed_scale:.2f}"
        )
        return linear_vel_limit, angular_vel_limit, detail

    def publish_cmd(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_vel_pub.publish(msg)

    def destroy_node(self):
        self.publish_cmd(0.0, 0.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OmniVLAEdgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()