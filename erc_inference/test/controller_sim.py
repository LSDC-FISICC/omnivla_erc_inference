#!/usr/bin/env python3
"""Closed-loop simulation of a checkpoint mission on a rover model fitted to field bags.

Runs the code the robot runs, not a copy of it:
- motion_control.MotionController: the control law inside omnivla_edge_node, fed
  with config/controller.yaml over motion_control.DEFAULT_PARAMS.
- motion_control.CommandShaper: the acceleration limits checkpoint_controller_node
  applies to /cmd_vel, with that node's declared defaults.
- checkpoint_controller_node's route geometry (route_to_local, project_forward,
  point_at) and its carrot, arrival and rate parameters.
- motion_control.robot_frame_offset on UTM deltas: the goal bearing exactly as
  omnivla_edge_node computes it.

What is modelled rather than run, and where each number comes from:
- Rover (RoverModel): command delay, first-order lags and gains from Tarea B and
  mission_16sept (k_v 1.11; k_w 1.18 in place, 0.36 median moving with IQR
  0.16-1.42); breakaway speeds below which the rover does not move (mission_16sept:
  held 0.049-0.065 m/s never moved, held 0.3 always did; the threshold in between
  is unmeasured, so it is swept).
- Localization: AR(1) position error 0.2 m / 3 s (EKF p50 0.19 m), heading bias
  +-6 deg (magnetometer +5.6 deg on mission9sept), 2 deg noise. Timing under both
  readings of mission_16sept (RoverModel.telemetry_latency_s): either 0.25 s
  sensor latency behind 1.3 s of actuation, or heading and speed 0.9-1.4 s old,
  dead-reckoned to now as ekf_global does, behind 0.1-0.4 s of actuation.
- The model (ModelEmulator), only needed by steering_source "model" and linear
  law "rho": fitted on mission_16sept's 683 polar ticks -- alpha = 0.117 x carrot
  bearing (r = 0.60), 5 deg noise correlated over ~2 s, and short-waypoint episodes
  (rho = 0.1 m) up to 19 s long.
- The SDK accepts every arrival after a 1.5 s confirmation.

Needs ROS 2 and a workspace providing erc_inference_msgs / erc_static_map_msgs
sourced (checkpoint_controller_node imports them), e.g.
    source /opt/ros/jazzy/setup.bash && source ~/lsdc_ws/install/setup.bash
    python3 test/controller_sim.py            # from erc_inference/; full comparison table
"""

import argparse
import ast
import importlib.util
import itertools
import math
import os
import re
import sys
from collections import deque
from dataclasses import dataclass, field, replace
from multiprocessing import Pool

import numpy as np
import utm
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
# The source tree, not whatever erc_inference an installed workspace provides.
sys.path.insert(0, PACKAGE_ROOT)

from erc_inference import motion_control as mc  # noqa: E402
from erc_inference import local_planner as lp  # noqa: E402

NODE_PATH = os.path.join(PACKAGE_ROOT, "erc_inference", "checkpoint_controller_node.py")
CONFIG_PATH = os.path.join(PACKAGE_ROOT, "config", "controller.yaml")
# omnivla_edge_node constants the simulation must match.
GOAL_RESET_DISTANCE_M = 3.0
THRES_DIST = 30.0


def load_checkpoint_node():
    spec = importlib.util.spec_from_file_location("checkpoint_controller_node", NODE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_node_defaults():
    """Parameter defaults as declared in checkpoint_controller_node.py."""
    source = open(NODE_PATH).read()
    return {name: ast.literal_eval(value)
            for name, value in re.findall(r"declare_parameter\('(\w+)', ([^)]+)\)", source)}


def controller_params(overrides=None):
    """DEFAULT_PARAMS, then config/controller.yaml, then overrides."""
    params = dict(mc.DEFAULT_PARAMS)

    def flatten(prefix, node):
        for key, value in node.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                flatten(name + ".", value)
            else:
                params[name] = value

    config = yaml.safe_load(open(CONFIG_PATH))
    flatten("", config["/**"]["ros__parameters"])
    params.update(overrides or {})
    unknown = set(params) - set(mc.DEFAULT_PARAMS)
    assert not unknown, f"parameters the node does not declare: {sorted(unknown)}"
    errors = mc.parameter_errors(params)
    assert not errors, errors
    return params


# ---------------------------------------------------------------------------
# Rover and sensors
# ---------------------------------------------------------------------------

import obstacles as obs


@dataclass
class RoverModel:
    delay_s: float = 1.3
    tau_v: float = 0.47
    tau_w: float = 0.35
    k_v: float = 1.11
    k_w_in_place: float = 1.18
    k_w_moving: float = 0.36
    v_breakaway: float = 0.15
    w_breakaway: float = 0.15
    heading_bias_deg: float = 6.0
    position_sigma_m: float = 0.2
    position_tau_s: float = 3.0
    heading_sigma_deg: float = 2.0
    # Two readings of mission_16sept's timing, which the bag cannot tell apart
    # (see docs/Mission16sept.md, "Latencias"). IMU, wheel and magnetometer
    # samples reach the bridge 1.16 s (p50) after their own timestamps, and the
    # command->gyro delay measured on bag time is ~1.3 s.
    #   telemetry_latency_s == 0: the timestamps are off by a clock offset; the
    #     data is only sensor_latency_s old and the 1.3 s is actuation (delay_s).
    #   telemetry_latency_s > 0: the data really is that old; the estimate is the
    #     state telemetry_latency_s ago dead-reckoned to now with that stale
    #     speed and heading (ekf_global with predict_to_current_time and no
    #     smooth_lagged_data), and delay_s is only the remaining actuation.
    sensor_latency_s: float = 0.25
    telemetry_latency_s: float = 0.0
    # heading_node holds its last output when the magnetometer drops out
    # (mission_16sept: holds of 5.2 s and 15.2 s). 0 = never.
    heading_freeze_every_s: float = 0.0
    heading_freeze_s: float = 5.0
    # The emulated model (ModelEmulator): the share of the carrot's bearing its
    # plan turns by (0.117 fitted on mission_16sept), and optional sidesteps --
    # every avoid_every_s the plan deviates avoid_deg beyond that for avoid_s,
    # alternating sides, standing in for going around something. 0 = never.
    # Field effects the default plant leaves out, all off by default so every
    # earlier result stands (rover_simulation/weave_check.py, mission_wuhan_hard):
    #   heading_extra_latency_s  the compass reaches the controller older than the
    #                            EKF position, which is extrapolated to now
    #   heading_hold_s           telemetry arrives in bursts every ~0.5 s, so the
    #                            heading steps instead of moving continuously
    #   k_w_moving_sigma         moving yaw gain varies (IQR 0.16-1.42 in the
    #                            field): log-normal multiplier, redrawn every k_w_tau_s
    heading_extra_latency_s: float = 0.0
    heading_hold_s: float = 0.0
    k_w_moving_sigma: float = 0.0
    k_w_tau_s: float = 2.0
    model_bearing_gain: float = 0.117
    avoid_every_s: float = 0.0
    avoid_s: float = 6.0
    avoid_deg: float = 25.0


class Rover:
    """Unicycle with delayed, gain-scaled, first-order actuation and breakaway speeds."""

    def __init__(self, model: RoverModel, x, y, theta, rng=None):
        self.m = model
        self.x, self.y, self.theta = x, y, theta
        self.v = 0.0
        self.w = 0.0
        self._commands = deque([(-math.inf, 0.0, 0.0)])
        self._rng = rng
        self._kw_mult = 1.0
        self._kw_next = 0.0

    def command(self, t, linear, angular):
        self._commands.append((t, linear, angular))

    def step(self, t, dt):
        while len(self._commands) > 1 and self._commands[1][0] <= t - self.m.delay_s:
            self._commands.popleft()
        _, linear, angular = self._commands[0]
        target_v = self.m.k_v * linear if abs(linear) >= self.m.v_breakaway else 0.0
        moving = abs(self.v) > 0.03 or target_v != 0.0
        if self.m.k_w_moving_sigma > 0.0 and self._rng is not None and t >= self._kw_next:
            sg = self.m.k_w_moving_sigma
            self._kw_mult = math.exp(sg * self._rng.standard_normal() - 0.5 * sg * sg)
            self._kw_next = t + self.m.k_w_tau_s
        if moving:
            target_w = self.m.k_w_moving * self._kw_mult * angular
        else:
            target_w = self.m.k_w_in_place * angular if abs(angular) >= self.m.w_breakaway else 0.0
        self.v += (target_v - self.v) * min(1.0, dt / self.m.tau_v)
        self.w += (target_w - self.w) * min(1.0, dt / self.m.tau_w)
        self.theta = mc.clip_angle(self.theta + self.w * dt)
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt


class Localization:
    """EKF-like position and magnetometer-like heading, as the nodes receive them."""

    def __init__(self, model: RoverModel, rng):
        self.m = model
        self.rng = rng
        self.err = np.zeros(2)
        self._history = deque()
        self._headings = deque()      # (t, heading) for heading_extra_latency_s
        self._held = None
        self._held_t = -math.inf

    def update(self, t, dt, rover: Rover):
        a = math.exp(-dt / self.m.position_tau_s)
        self.err = a * self.err + math.sqrt(1 - a * a) * self.m.position_sigma_m * self.rng.standard_normal(2)
        heading = rover.theta + math.radians(self.m.heading_bias_deg + self.m.heading_sigma_deg
                                             * self.rng.standard_normal())
        if (self.m.heading_freeze_every_s > 0.0 and self._history
                and t % self.m.heading_freeze_every_s < self.m.heading_freeze_s):
            heading = self._history[-1][3]
        self._history.append((t, rover.x + self.err[0], rover.y + self.err[1], heading, rover.v))
        self._now = t
        if self.m.heading_extra_latency_s > 0.0:
            self._headings.append((t, heading))
            base = self.m.telemetry_latency_s if self.m.telemetry_latency_s > 0.0 else self.m.sensor_latency_s
            while len(self._headings) > 1 and self._headings[1][0] <= t - base - self.m.heading_extra_latency_s:
                self._headings.popleft()
        latency = self.m.telemetry_latency_s if self.m.telemetry_latency_s > 0.0 else self.m.sensor_latency_s
        while len(self._history) > 1 and self._history[1][0] <= t - latency:
            self._history.popleft()

    def _heading(self, theta):
        if self.m.heading_extra_latency_s > 0.0 and self._headings:
            theta = self._headings[0][1]
        if self.m.heading_hold_s > 0.0:
            if self._held is None or self._now - self._held_t >= self.m.heading_hold_s:
                self._held, self._held_t = theta, self._now
            theta = self._held
        return theta

    def estimate(self):
        _, x, y, theta, v = self._history[0]
        if self.m.heading_extra_latency_s > 0.0 or self.m.heading_hold_s > 0.0:
            theta = self._heading(theta)
        lag = self.m.telemetry_latency_s
        if lag > 0.0:
            x += v * lag * math.cos(theta)
            y += v * lag * math.sin(theta)
        return x, y, theta


# ---------------------------------------------------------------------------
# The model's waypoint
# ---------------------------------------------------------------------------

def avoid_window(model: RoverModel, t):
    """Index of the sidestep window t falls in, or None."""
    if model.avoid_every_s <= 0.0 or t < model.avoid_every_s:
        return None
    if t % model.avoid_every_s >= model.avoid_s:
        return None
    return int(t // model.avoid_every_s)


class ModelEmulator:
    NOISE_DEG = 5.0
    NOISE_TAU_S = 2.0
    ALPHA_LIMIT_DEG = 25.0
    SHORT_RATE_PER_S = 1.0 / 40.0
    SHORT_MEAN_S = 7.0

    def __init__(self, rng, model: RoverModel):
        self.rng = rng
        self.m = model
        self.noise = 0.0
        self.short_until = -1.0

    def chunk(self, t, dt, carrot_bearing):
        """The model's 8 waypoints: a straight path at the emulated bearing, index 4 at rho.

        Without sidesteps (RoverModel.avoid_every_s = 0) the emulation has no
        obstacles in it, so it shows what giving the model steering authority costs
        in tracking, not what it buys in avoidance.
        """
        a = math.exp(-dt / self.NOISE_TAU_S)
        self.noise = a * self.noise + math.sqrt(1 - a * a) * self.NOISE_DEG * self.rng.standard_normal()
        if t >= self.short_until and self.rng.random() < self.SHORT_RATE_PER_S * dt:
            self.short_until = t + self.rng.exponential(self.SHORT_MEAN_S)
        alpha_deg = float(np.clip(self.m.model_bearing_gain * math.degrees(carrot_bearing) + self.noise,
                                  -self.ALPHA_LIMIT_DEG, self.ALPHA_LIMIT_DEG))
        window = avoid_window(self.m, t)
        if window is not None:
            alpha_deg += self.m.avoid_deg if window % 2 == 0 else -self.m.avoid_deg
        alpha = math.radians(alpha_deg)
        rho = 0.1 if t < self.short_until else float(np.clip(1.4 + 0.15 * self.rng.standard_normal(), 0.3, 1.7))
        return [(rho * (i + 1) / 5 * math.cos(alpha), rho * (i + 1) / 5 * math.sin(alpha),
                 math.cos(alpha), math.sin(alpha)) for i in range(8)]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

LAT0, LON0 = 30.4825077, 114.3026199
K_NORTH = math.radians(1.0) * 6378137.0
K_EAST = K_NORTH * math.cos(math.radians(LAT0))


def to_latlon(east, north):
    return LAT0 + north / K_NORTH, LON0 + east / K_EAST


def to_local(lat, lon):
    return (lon - LON0) * K_EAST, (lat - LAT0) * K_NORTH


@dataclass
class Scenario:
    name: str
    start: tuple            # (east, north, heading ENU rad)
    legs: list              # each leg: list of (east, north) route points, the last is the checkpoint
    # Things the A* route does not know about, because OSM does not have kerbs,
    # steps or parked bicycles. Empty means the clear-path world every scenario
    # assumed before obstacle avoidance existed.
    obstacles: list = field(default_factory=list)


def mission_16sept_legs():
    cp13 = to_local(30.48243713, 114.30264280)
    cp2 = to_local(30.48268318, 114.30260470)
    return [[cp13], [cp2], [cp13]]


SCENARIOS = {
    "straight": Scenario("straight", (0.0, 0.0, 0.0), [[(30.0, 0.0)]]),
    "L-turn": Scenario("L-turn", (0.0, 0.0, 0.0), [[(15.0, 0.0), (15.0, 15.0)]]),
    "U-turn": Scenario("U-turn", (0.0, 0.0, 0.0), [[(-25.0, 0.0)]]),
    "zigzag": Scenario("zigzag", (0.0, 0.0, 0.0), [[(10.0, 0.0), (16.0, 8.0), (26.0, 8.0), (32.0, 0.0)]]),
    # mission_16sept's checkpoints, starting where the bag started (rover facing
    # roughly away from checkpoint 1, as in leg 3 of that bag).
    "mission_16sept": Scenario("mission_16sept", (0.0, 0.0, math.radians(90.0)), mission_16sept_legs()),
    # --- worlds with things A* cannot see -------------------------------
    # A kerb across the route with a gap on one side: the 16-sept case, where
    # OSM showed open plaza and a 15-20 cm granite edge was waiting.
    "kerb": Scenario("kerb", (0.0, 0.0, 0.0), [[(20.0, 0.0)]],
                     [obs.Segment((8.0, -3.0), (8.0, 0.55))]),
    # Something standing in the middle of the route: a post, a bin, another
    # rover. Either side works, which is where a selector can dither.
    "post": Scenario("post", (0.0, 0.0, 0.0), [[(20.0, 0.0)]],
                     [obs.Disc((8.0, 0.0), 0.40)]),
    # Two offset obstacles: clearing the first points the rover at the second.
    # This is the one that punishes reacting to only the current frame.
    "chicane": Scenario("chicane", (0.0, 0.0, 0.0), [[(24.0, 0.0)]],
                        [obs.Segment((8.0, -3.0), (8.0, 0.55)),
                         obs.Segment((13.0, -0.55), (13.0, 3.0))]),
    # The 22-sept hedge: a long planter wall crossing the route at a shallow
    # angle, so the carrot keeps pulling the rover back into it after every
    # side-step (mission_carrot_sidestep*: 5 brakes 2-2.5 m apart along it).
    "hedge": Scenario("hedge", (0.0, 0.0, 0.0), [[(26.0, 0.0)]],
                      [obs.Segment((4.0, -4.0), (20.0, 1.2))]),
    # A raised planter sitting on the route, wider than the camera sees at once.
    "planter": Scenario("planter", (0.0, 0.0, 0.0), [[(22.0, 0.0)]],
                        [obs.Segment((8.5, -2.5), (11.5, -2.5)), obs.Segment((11.5, -2.5), (11.5, 2.5)),
                         obs.Segment((11.5, 2.5), (8.5, 2.5)), obs.Segment((8.5, 2.5), (8.5, -2.5))]),
    # Stairs or a long wall across the route, passable only past one end.
    "wall": Scenario("wall", (0.0, 0.0, 0.0), [[(20.0, 0.0)]],
                     [obs.Segment((9.0, -6.0), (9.0, 2.0))]),
}


# ---------------------------------------------------------------------------
# Mission loop
# ---------------------------------------------------------------------------

@dataclass
class Result:
    completed: bool
    time_s: float
    cross_track_p50: float
    cross_track_p95: float
    cross_track_max: float
    stalled_s: float
    dead_command_s: float
    max_linear_step: float
    max_angular_step: float
    simultaneous_steps: int
    weave_per_min: float
    max_lateral_accel: float
    goal_turns: int
    arrival_error_max: float
    # With obstacles in the world, not hitting them is a harder
    # requirement than tracking the route, and a separate one.
    hit: bool = False
    min_clearance_m: float = float('inf')
    # local_planner.LocalReplanner: how many times the route was spliced
    replans: int = 0
    # Median over the model's sidestep windows of how far the rover moved toward
    # the side the model asked for, from where it was when the window opened
    # (signed distance to the route); nan without sidesteps.
    sidestep_m: float = float("nan")
    trace: dict = field(default_factory=dict, repr=False)


def signed_route_offset(p, segments):
    """Distance to the closest route segment, positive to the left of travel."""
    best, signed = math.inf, 0.0
    for a, b in segments:
        d = point_segment_distance(p, a, b)
        if d < best:
            ab = b - a
            side = ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])
            best, signed = d, math.copysign(d, side) if side != 0.0 else 0.0
    return signed


def point_segment_distance(p, a, b):
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.hypot(*(p - (a + t * ab))))



def plan_for_deviation(delta, goal_bearing, params):
    """A waypoint chunk whose plan yields exactly `delta` of steering deviation.

    The avoidance deviation is pushed through the SAME code path the model's plan
    uses, so it meets the real deadzone, the real cap and the real delay
    compensation rather than a reimplementation of them. MotionController
    computes

        raw       = plan_bearing - plan_bearing_gain * goal_bearing
        deviation = clip(sign(raw) * max(0, |raw| - deadzone), +-cap)

    so inverting it means adding the deadzone back before handing it over.

    NOTE for the real node: feed the deviation directly instead of through this
    channel. The 8 deg deadzone was tuned to swallow the model's own ~5 deg of
    bearing noise; an avoidance command is not noise and should not have to pay
    it. Compensating here keeps the simulator honest about everything else.
    """
    gain = params["polar.plan_bearing_gain"]
    dz = math.radians(params["polar.plan_deviation_deadzone_deg"])
    d = float(delta)
    mag = abs(d) + dz if abs(d) > 1e-6 else 0.0
    plan_bearing = gain * goal_bearing + math.copysign(mag, d if d else 1.0)
    # straight path at that bearing, long enough to clear plan_min_length_m
    step = 0.4
    return [(step * (i + 1) * math.cos(plan_bearing),
             step * (i + 1) * math.sin(plan_bearing), 0.0, 0.0) for i in range(8)]


def simulate(scenario: Scenario, params: dict, model: RoverModel, seed: int, record=False,
             dt=0.02, node=None, node_defaults=None, avoid=None, stop=None, override=None,
             local=None):
    """local: None, or a dict of local_planner parameters -> a LocalReplanner per leg,
    fed the perception profile at the carrot rate, exactly where
    checkpoint_controller_node runs it."""
    node = node or load_checkpoint_node()
    cp = node_defaults or checkpoint_node_defaults()
    rng = np.random.default_rng(seed)
    rover = Rover(model, *scenario.start, rng=np.random.default_rng(seed + 1000))
    loc = Localization(model, rng)
    emulator = ModelEmulator(rng, model)
    controller = mc.MotionController(params)
    shaper = mc.CommandShaper(cp["max_linear_accel"], cp["max_linear_decel"],
                              cp["max_angular_accel"], cp["max_angular_decel"])

    route_len = sum(float(np.hypot(*np.diff(np.array([scenario.start[:2]] + leg), axis=0).T).sum())
                    for leg in scenario.legs)
    t_limit = 4.0 * route_len / params["max_linear_vel"] + 60.0

    tick_period = 1.0 / params["tick_rate"]
    inference_latency = 0.17
    output_period = 1.0 / cp["control_rate_hz"]
    carrot_period = 1.0 / cp["carrot_rate_hz"]
    confirm_s = 1.5

    # Checkpoint controller state
    leg_index = 0
    phase = "start_leg"
    confirm_until = 0.0
    motion_allowed = False
    route = None
    s_proj = 0.0
    last_carrot = -math.inf
    started = False
    arrival_errors = []
    # Edge node state
    enabled = False
    goal = None
    next_tick = 0.0
    pending_cmd = None
    model_cmd = (0.0, 0.0)
    # Output state
    next_output = 0.0
    out = (0.0, 0.0)
    goal_turns = 0
    was_turning = False
    hit = False
    min_clearance = float('inf')
    world = list(getattr(scenario, 'obstacles', []) or [])
    planner = None
    replans = 0
    local_notes = []

    log_t, log_out, log_true, cross, route_segments = [], [], [], [], []
    stop_resets = set()   # output indices that follow an immediate stop
    sidestep = {}         # sidestep window -> (offset at its start, best progress toward the asked side)
    ticks = []            # (t, leg, mode, carrot bearing, model alpha, linear, angular)
    leg_times = []
    t = 0.0
    loc.update(t, dt, rover)
    while t < t_limit:
        loc.update(t, dt, rover)
        ex, ey, etheta = loc.estimate()
        est_lat, est_lon = to_latlon(ex, ey)

        # ---- checkpoint_controller_node ----
        if phase == "start_leg":
            leg = scenario.legs[leg_index]
            checkpoint = leg[-1]
            points = [(ex, ey)] + list(leg)
            route = [to_latlon(*p) for p in points]
            frame, pts, cum = node.route_to_local(route)
            goal_e, goal_n = node.latlon_to_local(frame, *to_latlon(*checkpoint))
            route_segments = [(np.array(a), np.array(b)) for a, b in zip(points, points[1:])]
            s_proj, started, last_carrot = 0.0, False, -math.inf
            leg_started = t
            planner = lp.LocalReplanner(pts, **local) if local is not None else None
            phase = "follow"
        if phase == "follow":
            east, north = node.latlon_to_local(frame, est_lat, est_lon)
            remaining = math.hypot(goal_e - east, goal_n - north)
            if remaining <= cp["checkpoint_proximity_m"]:
                arrival_errors.append(math.hypot(rover.x - checkpoint[0], rover.y - checkpoint[1]))
                # _request_stop: motion off, immediate zero, inference disabled
                motion_allowed = False
                shaper.reset()
                out = (0.0, 0.0)
                rover.command(t, 0.0, 0.0)
                stop_resets.add(len(log_out))
                if enabled:
                    enabled = False
                    controller.reset()
                phase, confirm_until = "confirm", t + confirm_s
                leg_times.append(t - leg_started)
            else:
                s_proj = node.project_forward(pts, cum, east, north, s_proj)
                if t - last_carrot >= carrot_period:
                    last_carrot = t
                    if planner is not None and world:
                        # the map is built from the ESTIMATED pose, perception
                        # from the true one -- as on the rover
                        pb, pf, pc = obs.profile(rover.x, rover.y, rover.theta, world, rng)
                        planner.observe(t, east, north, etheta, pb, pf, pf < pc - 1e-3)
                        new_pts, note = planner.check(t, east, north, pts, cum, s_proj)
                        if note:
                            local_notes.append((round(t, 1), note))
                        if new_pts is not None:
                            pts = new_pts
                            cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
                            s_proj = 0.0
                            replans += 1
                    ce, cn, _ = node.point_at(pts, cum, s_proj + cp["carrot_distance_m"])
                    new_goal = node.local_to_latlon(frame, ce, cn)
                    # goal_gps_callback: a jump means a new target
                    if goal is None or math.hypot(*np.subtract(to_local(*new_goal), to_local(*goal))) > GOAL_RESET_DISTANCE_M:
                        controller.reset()
                    goal = new_goal
                    if not started:
                        started = True
                        motion_allowed = True
                        if not enabled:
                            enabled = True
                            controller.reset()
        elif phase == "confirm" and t >= confirm_until:
            leg_index += 1
            if leg_index == len(scenario.legs):
                break
            phase = "start_leg"

        # ---- omnivla_edge_node tick ----
        if t >= next_tick:
            next_tick += tick_period
            if not enabled or goal is None:
                model_cmd = (0.0, 0.0)
                controller.note_idle(t)
            else:
                cur = utm.from_latlon(est_lat, est_lon)
                gl = utm.from_latlon(*goal)
                heading_deg = 90.0 - math.degrees(etheta)
                rel_x, rel_y = mc.robot_frame_offset(gl[0] - cur[0], gl[1] - cur[1], heading_deg)
                bearing = mc.goal_bearing_from_offset(rel_x, rel_y)
                radius = math.hypot(rel_x, rel_y)
                chunk = emulator.chunk(t, tick_period, bearing)
                if avoid is not None and world:
                    # Perception sees the world from where the rover actually is.
                    bearings_deg, free, caps = obs.profile(rover.x, rover.y, rover.theta,
                                                           world, rng)
                    delta = avoid(bearings_deg, free, bearing, caps)
                    chunk = plan_for_deviation(delta, bearing, params)
                waypoint = chunk[4]
                mode, v, w, _ = controller.command(t + inference_latency, params, bearing, radius,
                                                   chunk, 4, True)
                if override is not None and world:
                    # erc_inference.sidestep.SideStep: fed estimated heading and
                    # position, as the node would be; perception from the truth
                    ob, of, _oc = obs.profile(rover.x, rover.y, rover.theta, world, rng)
                    v, w, _note, resumed = override.step(t, etheta, (cur[0], cur[1]),
                                                         ob, of, v, w, bearing)
                    if resumed:
                        controller.reset()
                if stop is not None and world and v > 0.0:
                    # carrot_controller_node's obstacle_stop: brake, never steer
                    sb, sf, _sc = obs.profile(rover.x, rover.y, rover.theta, world, rng)
                    if stop(sb, sf):
                        v, w = 0.0, 0.0
                ticks.append((t, leg_index, mode, bearing, math.atan2(waypoint[1], waypoint[0]), v, w))
                turning = mode == "goal-turn"
                goal_turns += int(turning and not was_turning)
                was_turning = turning
                pending_cmd = (t + inference_latency, (v, w))
        if pending_cmd is not None and t >= pending_cmd[0]:
            model_cmd = pending_cmd[1]
            pending_cmd = None

        # ---- checkpoint_controller_node output timer ----
        if t >= next_output:
            next_output += output_period
            target = model_cmd if motion_allowed else (0.0, 0.0)
            out = shaper.step(target[0], target[1], output_period)
            rover.command(t, *out)
            log_t.append(t)
            log_out.append(out)
            log_true.append((rover.v, rover.w, rover.x, rover.y))
            if phase == "follow" and route_segments:
                p = np.array([rover.x, rover.y])
                d = min(point_segment_distance(p, a, b) for a, b in route_segments)
                cross.append(d)
                # Judged on the rover's own motion: the 1.3 s delay lands the
                # sidestep after its window, so windows extend by 3 s.
                w = avoid_window(model, t) if avoid_window(model, t) is not None else avoid_window(model, t - 3.0)
                if w is not None:
                    side = 1.0 if w % 2 == 0 else -1.0   # matches ModelEmulator: even windows go left
                    offset = side * signed_route_offset(p, route_segments)
                    start, best = sidestep.get(w, (offset, 0.0))
                    sidestep[w] = (start, max(best, offset - start))

        rover.step(t, dt)
        if world:
            # the rover really is where it is, not where the EKF thinks;
            # a collision does not care about localisation error
            c = obs.clearance((rover.x, rover.y), world)
            min_clearance = min(min_clearance, c)
            hit = hit or c <= obs.FOOTPRINT_W / 2
        t += dt

    completed = leg_index == len(scenario.legs)
    out_arr = np.array(log_out) if log_out else np.zeros((1, 2))
    true_arr = np.array(log_true) if log_true else np.zeros((1, 4))
    T = np.array(log_t) if log_t else np.zeros(1)
    # Stalled: the command the rover is acting on now asks for motion, nothing moves.
    delayed_idx = np.clip(np.searchsorted(T, T - model.delay_s) - 1, 0, None)
    acting = out_arr[delayed_idx]
    asks = (np.abs(acting[:, 0]) > 0.02) | (np.abs(acting[:, 1]) > 0.02)
    still = (np.abs(true_arr[:, 0]) < 0.02) & (np.abs(true_arr[:, 1]) < 0.02)
    stalled_s = float(np.sum(asks & still) * output_period)
    dead = ((np.abs(out_arr[:, 0]) > 0.0) & (np.abs(out_arr[:, 0]) < model.v_breakaway)
            & (np.abs(out_arr[:, 1]) < model.w_breakaway))
    steps = np.abs(np.diff(out_arr, axis=0)) if len(out_arr) > 1 else np.zeros((1, 2))
    # A checkpoint stop is an immediate zero by design; only shaped changes count.
    for index in stop_resets:
        if 0 < index <= len(steps):
            steps[index - 1] = 0.0
    w_true = true_arr[:, 1]
    significant = np.abs(w_true) > 0.08
    signs = np.sign(w_true[significant])
    weave = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    minutes = max(T[-1], 1.0) / 60.0
    result = Result(
        completed=completed,
        time_s=float(t),
        cross_track_p50=float(np.percentile(cross, 50)) if cross else float("nan"),
        cross_track_p95=float(np.percentile(cross, 95)) if cross else float("nan"),
        cross_track_max=float(np.max(cross)) if cross else float("nan"),
        stalled_s=stalled_s,
        dead_command_s=float(np.sum(dead) * output_period),
        max_linear_step=float(steps[:, 0].max() / output_period),
        max_angular_step=float(steps[:, 1].max() / output_period),
        simultaneous_steps=int(np.sum((steps[:, 0] > 0.1) & (steps[:, 1] > 0.1))),
        weave_per_min=weave / minutes,
        max_lateral_accel=float(np.max(np.abs(true_arr[:, 0] * true_arr[:, 1]))),
        goal_turns=goal_turns,
        arrival_error_max=float(max(arrival_errors)) if arrival_errors else float("nan"),
        sidestep_m=float(np.median([best for _, best in sidestep.values()])) if sidestep else float("nan"),
        hit=hit,
        min_clearance_m=min_clearance,
        replans=replans,
    )
    result.trace = dict(local_notes=local_notes, planner=planner)
    if record:
        result.trace.update(t=T, out=out_arr, true=true_arr, ticks=ticks, leg_times=leg_times)
    return result


# ---------------------------------------------------------------------------
# Controllers compared and the robustness sweep
# ---------------------------------------------------------------------------

# config/controller.yaml as it was at af980ba, the build that drove mission_16sept.
AS_DEPLOYED_16SEPT = {
    "polar.steering_source": "model",
    "polar.linear_law": "rho",
    "polar.k_rho": 0.5,
    "polar.k_alpha": 1.5,
    "polar.k_beta": 0.0,
    "goal_turn.exit_deg": 30.0,
    "goal_turn.brake_s": 0.0,
}
NO_SHAPING = {"max_linear_accel": 0.0, "max_linear_decel": 0.0, "max_angular_accel": 0.0, "max_angular_decel": 0.0}

# One grid per reading of mission_16sept's timing (see RoverModel). Both are
# swept because the bag cannot say which one is true; results break down by
# telemetry_latency_s (0.0 = the clock-offset reading).
SWEEPS = {
    "clock offset: all 1.3 s is actuation": {
        "delay_s": [0.9, 1.3, 1.8],
        "k_w_moving": [0.2, 0.36, 1.0],
        "v_breakaway": [0.1, 0.2],
        "heading_bias_deg": [-6.0, 6.0],
    },
    "telemetry lag: data 1.16 s old, short actuation": {
        "telemetry_latency_s": [0.9, 1.16, 1.4],
        "delay_s": [0.1, 0.2, 0.4],
        "k_w_moving": [0.2, 0.36, 1.0],
        "v_breakaway": [0.1, 0.2],
        "heading_bias_deg": [-6.0, 6.0],
    },
}


def plant_grid():
    for sweep in SWEEPS.values():
        keys = list(sweep)
        for values in itertools.product(*(sweep[k] for k in keys)):
            yield replace(RoverModel(), **dict(zip(keys, values)))


_WORKER = {}


def _run(job):
    scenario_name, param_overrides, shaping_overrides, model, seed = job
    if "node" not in _WORKER:
        _WORKER["node"] = load_checkpoint_node()
        _WORKER["defaults"] = checkpoint_node_defaults()
    defaults = dict(_WORKER["defaults"])
    defaults.update(shaping_overrides)
    params = controller_params(param_overrides)
    return scenario_name, model, simulate(SCENARIOS[scenario_name], params, model, seed,
                                          node=_WORKER["node"], node_defaults=defaults)


def evaluate(param_overrides, shaping_overrides=None, scenarios=None, seeds=(0, 1), processes=None):
    jobs = [(name, param_overrides, shaping_overrides or {}, model, seed)
            for name in (scenarios or SCENARIOS)
            for model in plant_grid()
            for seed in seeds]
    with Pool(processes) as pool:
        return pool.map(_run, jobs, chunksize=4)


def summarize(label, results):
    R = [r for _, _, r in results]
    completed = np.mean([r.completed for r in R])
    def pct(values, q):
        values = [v for v in values if not math.isnan(v)]
        return np.percentile(values, q) if values else float("nan")
    print(f"{label:34s} done {100 * completed:5.1f}% | time p50 {pct([r.time_s for r in R], 50):5.0f}s | "
          f"xtrack p50 {pct([r.cross_track_p50 for r in R], 50):4.2f} p95 {pct([r.cross_track_p95 for r in R], 95):4.2f} "
          f"max {pct([r.cross_track_max for r in R], 100):4.1f}m | stalled p95 {pct([r.stalled_s for r in R], 95):5.1f}s "
          f"| dead-cmd p95 {pct([r.dead_command_s for r in R], 95):5.1f}s | dv/dt max {max(r.max_linear_step for r in R):4.1f} "
          f"dw/dt max {max(r.max_angular_step for r in R):4.1f} | v+w steps {sum(r.simultaneous_steps for r in R):4d} "
          f"| weave p95 {pct([r.weave_per_min for r in R], 95):4.1f}/min | a_lat max {max(r.max_lateral_accel for r in R):.2f}")


def breakdown(results, key):
    groups = {}
    for name, model, r in results:
        k = name if key == "scenario" else getattr(model, key)
        groups.setdefault(k, []).append(r)
    for k, R in groups.items():
        failed = [r for r in R if not r.completed]
        print(f"    {key}={k!s:15s} done {100 * (1 - len(failed) / len(R)):5.1f}%  "
              f"time p50 {np.median([r.time_s for r in R]):5.0f}s  "
              f"xtrack p95 {np.nanpercentile([r.cross_track_p95 for r in R], 95):4.2f}m  "
              f"stalled p95 {np.percentile([r.stalled_s for r in R], 95):5.1f}s  "
              f"weave p95 {np.percentile([r.weave_per_min for r in R], 95):4.1f}/min")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--tune", action="store_true", help="sweep the new controller's gains")
    args = parser.parse_args()

    runs = {
        "as deployed 16-sept": (AS_DEPLOYED_16SEPT, NO_SHAPING),
        "16-sept + shaping + brake": ({**AS_DEPLOYED_16SEPT, "goal_turn.brake_s": 1.0}, {}),
        "model steering + cruise": ({"polar.steering_source": "model"}, {}),
        "carrot steering + cruise": ({"polar.steering_source": "carrot"}, {}),
        "plan within a cone of the carrot (config)": ({}, {}),
    }
    if args.tune:
        runs = {}
        for k, dz, comp, exit_deg in itertools.product([0.5, 0.8, 1.2], [3.0], [0.0, 0.5], [30.0, 45.0]):
            runs[f"k={k} dz={dz} comp={comp} exit={exit_deg}"] = (
                {"polar.k_heading": k, "polar.heading_deadzone_deg": dz,
                 "polar.delay_compensation_gain": comp, "goal_turn.exit_deg": exit_deg}, {})

    for label, (overrides, shaping) in runs.items():
        results = evaluate(overrides, shaping, processes=args.processes)
        summarize(label, results)
        if not args.tune:
            breakdown(results, "scenario")
            breakdown(results, "telemetry_latency_s")
            breakdown(results, "v_breakaway")
            breakdown(results, "k_w_moving")
            breakdown(results, "delay_s")


if __name__ == "__main__":
    main()
