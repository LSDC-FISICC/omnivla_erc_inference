"""Unit tests for erc_inference.motion_control. Needs numpy and PyYAML, not ROS."""

import math
import os
import sys

import numpy as np
import pytest
import yaml

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

from erc_inference import motion_control as mc  # noqa: E402

CONFIG_PATH = os.path.join(PACKAGE_ROOT, "config", "controller.yaml")


def config_params():
    params = dict(mc.DEFAULT_PARAMS)

    def flatten(prefix, node):
        for key, value in node.items():
            if isinstance(value, dict):
                flatten(f"{prefix}{key}.", value)
            else:
                params[f"{prefix}{key}"] = value

    flatten("", yaml.safe_load(open(CONFIG_PATH))["/**"]["ros__parameters"])
    return params


def test_config_only_sets_declared_parameters():
    config = yaml.safe_load(open(CONFIG_PATH))["/**"]["ros__parameters"]
    names = set()

    def collect(prefix, node):
        for key, value in node.items():
            if isinstance(value, dict):
                collect(f"{prefix}{key}.", value)
            else:
                names.add(f"{prefix}{key}")

    collect("", config)
    assert names <= set(mc.DEFAULT_PARAMS), sorted(names - set(mc.DEFAULT_PARAMS))
    for name in names:
        assert type(config_params()[name]) is type(mc.DEFAULT_PARAMS[name]), name
    assert mc.parameter_errors(config_params()) == []


@pytest.mark.parametrize("bad", [
    {"controller_type": "mppi"},
    {"polar.steering_source": "gps"},
    {"polar.linear_law": "constant"},
    {"polar.min_linear_vel": 0.5},
    {"polar.slowdown_end_deg": 10.0},
])
def test_parameter_errors_reject_unusable_sets(bad):
    assert mc.parameter_errors({**mc.DEFAULT_PARAMS, **bad})


def test_cruise_speed_never_commands_below_breakaway_floor():
    for degrees in np.linspace(0.0, 180.0, 721):
        for sign in (1.0, -1.0):
            v = mc.cruise_speed(sign * math.radians(degrees), 0.3, 0.25, math.radians(30.0), math.radians(60.0))
            assert 0.25 <= v <= 0.3


def test_cruise_speed_tapers_between_start_and_end():
    start, end = math.radians(30.0), math.radians(60.0)
    assert mc.cruise_speed(math.radians(10.0), 0.3, 0.2, start, end) == pytest.approx(0.3)
    assert mc.cruise_speed(math.radians(45.0), 0.3, 0.2, start, end) == pytest.approx(0.25)
    assert mc.cruise_speed(math.radians(90.0), 0.3, 0.2, start, end) == pytest.approx(0.2)


def test_steer_deadzone_is_continuous_and_clipped():
    dz = math.radians(6.0)
    assert mc.steer(math.radians(5.9), 0.5, 0.3, dz) == 0.0
    assert mc.steer(math.radians(6.001), 0.5, 0.3, dz) == pytest.approx(0.0, abs=1e-4)
    assert mc.steer(math.radians(20.0), 0.5, 0.3, dz) == pytest.approx(0.5 * math.radians(14.0))
    assert mc.steer(-math.radians(20.0), 0.5, 0.3, dz) == pytest.approx(-0.5 * math.radians(14.0))
    assert mc.steer(math.radians(170.0), 0.5, 0.3, dz) == pytest.approx(0.3)
    assert mc.steer(-math.radians(170.0), 0.5, 0.3, dz) == pytest.approx(-0.3)


def test_slew_limiter_rates_and_zero_crossing():
    limiter = mc.SlewLimiter(accel=0.3, decel=0.6)
    values = [limiter.step(0.3, 0.1) for _ in range(12)]
    assert np.all(np.diff([0.0] + values) <= 0.03 + 1e-12)
    assert values[-1] == pytest.approx(0.3)
    values = [limiter.step(0.0, 0.1) for _ in range(6)]
    assert np.all(np.diff([0.3] + values) >= -0.06 - 1e-12)
    assert values[-1] == pytest.approx(0.0)

    angular = mc.SlewLimiter(accel=0.6, decel=1.2)
    angular.reset(0.3)
    previous = 0.3
    for _ in range(20):
        value = angular.step(-0.3, 0.1)
        # Toward zero at decel, away from it at accel, never faster than decel.
        assert abs(value - previous) <= 0.12 + 1e-12
        previous = value
    assert previous == pytest.approx(-0.3)


def test_slew_limiter_zero_rate_disables_limit():
    limiter = mc.SlewLimiter(accel=0.0, decel=0.0)
    assert limiter.step(0.3, 0.1) == 0.3
    assert limiter.step(-0.3, 0.1) == -0.3


def test_delay_compensator_integrates_recent_commands_only():
    comp = mc.DelayCompensator()
    for i in range(30):
        comp.record(i * 0.1, 0.3)
    assert comp.pending_turn(2.9, 1.3, 1.0) == pytest.approx(0.3 * 1.3, rel=1e-6)
    assert comp.pending_turn(2.9, 1.3, 0.5) == pytest.approx(0.5 * 0.3 * 1.3, rel=1e-6)
    assert comp.pending_turn(2.9, 1.3, 0.0) == 0.0
    comp.record(3.0, 0.0)
    assert comp.pending_turn(4.0, 1.3, 1.0) == pytest.approx(0.3 * 0.3, rel=1e-6)
    comp.reset()
    assert comp.pending_turn(4.0, 1.3, 1.0) == 0.0


def straight_chunk(rho, bearing=0.0):
    """8 waypoints on a straight path at `bearing`, index 4 at distance rho."""
    return [(rho * (i + 1) / 5 * math.cos(bearing), rho * (i + 1) / 5 * math.sin(bearing),
             math.cos(bearing), math.sin(bearing)) for i in range(8)]


def run_controller(params, bearings_deg, dt=1.0 / 3.0, distance=1.5):
    controller = mc.MotionController(params)
    out = []
    for i, b in enumerate(bearings_deg):
        out.append(controller.command(i * dt, params, math.radians(b), distance, straight_chunk(1.4), 4, True))
    return out


def test_goal_turn_brakes_before_spinning_and_latches_direction():
    params = config_params()
    out = run_controller(params, [20.0, 120.0, 150.0, 170.0, -175.0, 175.0, 100.0, 60.0, 40.0])
    modes = [m for m, _, _, _ in out]
    assert modes[0] == "polar"
    assert modes[1:8] == ["goal-turn"] * 7
    brake_ticks = int(math.ceil(params["goal_turn.brake_s"] * 3.0))
    for _, v, w, detail in out[1:1 + brake_ticks]:
        assert (v, w) == (0.0, 0.0) and "braking" in detail
    spinning = [w for _, _, w, _ in out[1 + brake_ticks:8]]
    assert spinning and all(w == pytest.approx(0.3) for w in spinning)
    assert modes[8] == "polar"


def test_carrot_polar_never_commands_a_speed_inside_the_floor():
    params = config_params()
    rng = np.random.default_rng(0)
    controller = mc.MotionController(params)
    for i in range(5000):
        bearing = math.radians(rng.uniform(-89.0, 89.0))
        dx, dy = rng.uniform(0.0, 1.7), rng.uniform(-0.5, 0.5)
        mode, v, w, _ = controller.command(i / 3.0, params, bearing, 1.5, [(dx, dy, 1.0, 0.0)] * 8, 4, True)
        assert mode == "polar"
        assert v == 0.0 or params["polar.min_linear_vel"] <= v <= params["max_linear_vel"]
        assert abs(w) <= params["max_angular_vel"]


def test_model_rho_law_reproduces_the_16sept_stall():
    params = {**config_params(), "polar.steering_source": "model", "polar.linear_law": "rho"}
    _, v, _, _ = mc.MotionController(params).command(0.0, params, 0.0, 1.5, straight_chunk(0.098), 4, True)
    assert v == pytest.approx(0.049)


def test_robot_frame_offset_matches_sdk_heading_convention():
    # Heading 90 (east): a goal 10 m east is straight ahead, one 10 m north is to the left.
    rel_x, rel_y = mc.robot_frame_offset(10.0, 0.0, 90.0)
    assert (rel_x, rel_y) == pytest.approx((0.0, 10.0), abs=1e-9)
    assert mc.goal_bearing_from_offset(*mc.robot_frame_offset(0.0, 10.0, 90.0)) == pytest.approx(math.pi / 2)


def test_plan_target_walks_the_path_and_handles_short_or_empty_plans():
    path = [(0.5 * (i + 1), 0.0, 1.0, 0.0) for i in range(4)] + [(2.0, 0.5 * (i + 1), 0.0, 1.0) for i in range(4)]
    bearing, length = mc.plan_target(path, 1.5)
    assert bearing == pytest.approx(0.0) and length == pytest.approx(4.0)
    bearing, _ = mc.plan_target(path, 3.0)
    assert bearing == pytest.approx(math.atan2(1.0, 2.0))
    bearing, length = mc.plan_target(straight_chunk(0.5, math.radians(20.0)), 1.5)
    assert bearing == pytest.approx(math.radians(20.0)) and length == pytest.approx(0.8)
    assert mc.plan_target([(0.0, 0.0, 1.0, 0.0)] * 8, 1.5) == (None, 0.0)


def test_plan_steering_is_bounded_by_the_carrot():
    params = {**config_params(), "polar.steering_source": "plan", "polar.max_plan_deviation_deg": 30.0,
              "polar.plan_bearing_gain": 1.0, "polar.plan_deviation_deadzone_deg": 0.0,
              "polar.delay_compensation_gain": 0.0, "polar.heading_deadzone_deg": 0.0}

    def bearing_steered(carrot_deg, plan_deg, rho=1.4):
        c = mc.MotionController(params)
        _, _, w, detail = c.command(0.0, params, math.radians(carrot_deg), 1.5,
                                    straight_chunk(rho, math.radians(plan_deg)), 4, True)
        return math.degrees(w / params["polar.k_heading"]), detail

    # Inside the cone the plan decides; outside it the plan is clipped toward the carrot.
    assert bearing_steered(10.0, -15.0)[0] == pytest.approx(-15.0, abs=1e-6)
    assert bearing_steered(40.0, 0.0)[0] == pytest.approx(10.0, abs=1e-6)
    assert bearing_steered(-40.0, 30.0)[0] == pytest.approx(-10.0, abs=1e-6)
    # A plan too short to trust leaves the carrot in charge.
    angle, detail = bearing_steered(20.0, -60.0, rho=0.1)
    assert angle == pytest.approx(20.0, abs=1e-6) and "plan=none" in detail
    # Cone 0 is pure carrot steering; cone 180 is pure plan.
    assert bearing_steered(25.0, -5.0)[0] != pytest.approx(25.0)
    params["polar.max_plan_deviation_deg"] = 0.0
    assert bearing_steered(25.0, -5.0)[0] == pytest.approx(25.0, abs=1e-6)


def test_plan_residual_removes_the_models_usual_underturning():
    params = {**config_params(), "polar.steering_source": "plan", "polar.plan_bearing_gain": 0.117,
              "polar.plan_deviation_deadzone_deg": 8.0, "polar.max_plan_deviation_deg": 30.0,
              "polar.delay_compensation_gain": 0.0, "polar.heading_deadzone_deg": 0.0}

    def bearing_steered(carrot_deg, plan_deg):
        c = mc.MotionController(params)
        _, _, w, _ = c.command(0.0, params, math.radians(carrot_deg), 1.5,
                               straight_chunk(1.4, math.radians(plan_deg)), 4, True)
        return math.degrees(w / params["polar.k_heading"])

    # The plan turning its usual 0.117x (plus noise inside the deadzone) leaves the carrot in charge.
    # (bearings kept under 34 deg: k_heading 0.5 saturates the 0.3 rad/s limit there)
    assert bearing_steered(20.0, 0.117 * 20.0) == pytest.approx(20.0, abs=1e-6)
    assert bearing_steered(20.0, 0.117 * 20.0 + 6.0) == pytest.approx(20.0, abs=1e-6)
    # Beyond it, the excess past the deadzone is added, up to the cone.
    assert bearing_steered(0.0, -25.0) == pytest.approx(-17.0, abs=1e-6)
    assert bearing_steered(0.0, 60.0) == pytest.approx(30.0, abs=1e-6)
