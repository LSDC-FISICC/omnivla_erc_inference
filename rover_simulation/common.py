"""Shared setup for the rover_simulation tools.

The simulator engine lives in ../erc_inference/test/ (controller_sim.py,
obstacles.py) because the package's pytest suite imports it from there; the
robot's own code it runs is ../erc_inference/erc_inference/. This puts both on
the path and wraps one simulated mission in a single call.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.normpath(os.path.join(HERE, '..', 'erc_inference'))
sys.path.insert(0, os.path.join(PACKAGE, 'test'))
sys.path.insert(0, PACKAGE)

import controller_sim as cs  # noqa: E402
import obstacles as obs  # noqa: E402
from erc_inference.sidestep import SideStep  # noqa: E402

# What each config name turns on, as mission_carrot.launch.py would:
#   carrot  carrot_controller_node alone (no reaction to obstacles)
#   ss      + obstacle_sidestep
#   lp      + checkpoint_controller_node local_replan
#   lp+ss   both (the field configuration)
CONFIGS = ('carrot', 'ss', 'lp', 'lp+ss')


def run(scenario, seed, config='lp+ss', carrot_m=1.5, ss=None, lp=None, model=None,
        record=False, override=None):
    """One mission. ss / lp: parameter overrides for SideStep / LocalReplanner.

    Returns (Result, SideStep or None). Result.trace['planner'] is the last leg's
    LocalReplanner, Result.trace['local_notes'] its log lines.
    """
    node = cs.load_checkpoint_node()
    defaults = dict(cs.checkpoint_node_defaults())
    defaults['carrot_distance_m'] = carrot_m
    params = cs.controller_params({'polar.steering_source': 'carrot'})
    sidestep = override if override is not None else (SideStep(**(ss or {})) if 'ss' in config else None)
    local = dict(lp or {}) if 'lp' in config else None
    result = cs.simulate(cs.SCENARIOS[scenario], params, model or cs.RoverModel(), seed=seed,
                         node=node, node_defaults=defaults, override=sidestep, local=local,
                         record=record)
    return result, sidestep
