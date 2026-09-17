"""Closed-loop regression: the configured controller must drive every scenario.

Uses test/controller_sim.py, which runs the robot's own control code against a
rover model fitted to mission_16sept (see that file for every number's source).

Skipped per-test, not at module level, where utm or checkpoint_controller_node's
message packages are not importable: `pytest.importorskip` / `pytest.skip(...,
allow_module_level=True)` here trips a bug in launch_testing_ros's pytest plugin
(registered by sourcing a ROS 2 environment) that discards every OTHER test file
passed to the same `pytest`/`colcon test` invocation -- reproduced with a two-file
minimal case, isolated to `launch_testing_ros_pytest_entrypoint.py`'s collection
hook. A runtime `pytestmark = pytest.mark.skipif(...)` does not hit that path.
"""

import os
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import controller_sim as cs
    _IMPORT_ERROR = None
except ImportError as exc:
    cs = None
    _IMPORT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    cs is None, reason=f"controller_sim's dependencies are not importable: {_IMPORT_ERROR}")


def _plants():
    if cs is None:
        return {}
    return {
        "nominal": cs.RoverModel(),
        "slow yaw, long delay": replace(cs.RoverModel(), k_w_moving=0.2, delay_s=1.8, v_breakaway=0.2),
        "fast yaw, short delay": replace(cs.RoverModel(), k_w_moving=1.0, delay_s=0.9, heading_bias_deg=-6.0),
        "heading holds": replace(cs.RoverModel(), heading_freeze_every_s=45.0, heading_freeze_s=15.0),
        # mission_16sept: telemetry 1.16 s old on arrival, the rest of the 1.3 s is actuation.
        "telemetry lag": replace(cs.RoverModel(), telemetry_latency_s=1.16, delay_s=0.2),
    }


PLANTS = _plants()


@pytest.fixture(scope="module")
def node():
    return cs.load_checkpoint_node(), cs.checkpoint_node_defaults()


@pytest.mark.parametrize("plant", list(PLANTS))
@pytest.mark.parametrize("scenario", list(cs.SCENARIOS) if cs else [])
def test_configured_controller_completes_without_stalls_or_steps(node, scenario, plant):
    module, defaults = node
    result = cs.simulate(cs.SCENARIOS[scenario], cs.controller_params(), PLANTS[plant], seed=0,
                         node=module, node_defaults=defaults)
    assert result.completed
    assert result.stalled_s < 2.0
    assert result.simultaneous_steps == 0
    assert result.max_linear_step <= max(defaults["max_linear_accel"], defaults["max_linear_decel"]) + 1e-6
    assert result.max_angular_step <= max(defaults["max_angular_accel"], defaults["max_angular_decel"]) + 1e-6
    assert result.cross_track_max < 3.5
    assert result.weave_per_min < 8.0


def test_deployed_16sept_controller_stalls_in_simulation(node):
    """The simulator must show the field failure, or it cannot vouch for the fix."""
    module, defaults = node
    defaults = {**defaults, **cs.NO_SHAPING}
    stalled = [cs.simulate(cs.SCENARIOS[name], cs.controller_params(cs.AS_DEPLOYED_16SEPT), cs.RoverModel(),
                           seed=0, node=module, node_defaults=defaults).stalled_s
               for name in cs.SCENARIOS]
    assert max(stalled) > 5.0


def test_plan_passes_the_models_sidesteps_and_carrot_does_not(node):
    """The plan's authority is the point of steering_source plan: a sidestep the model asks for reaches the rover.

    Judged on the straight route over several seeds: on routes with corners the
    signed offset jumps between segments and a single run says little.
    """
    module, defaults = node
    plant = replace(cs.RoverModel(), avoid_every_s=25.0, avoid_s=10.0)
    moved = {}
    for source in ("plan", "carrot"):
        moved[source] = sorted(cs.simulate(cs.SCENARIOS["straight"], cs.controller_params({"polar.steering_source": source}),
                                           plant, seed=seed, node=module, node_defaults=defaults).sidestep_m
                               for seed in range(6))
    median = {k: 0.5 * (v[2] + v[3]) for k, v in moved.items()}
    assert median["plan"] > 0.3
    assert median["carrot"] < 0.15
