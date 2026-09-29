#!/usr/bin/env python3
"""The NYU indoor loop in closed loop, on dead reckoning, between corridor walls.

controller_sim.simulate() as is -- MotionController + config/controller.yaml,
the carrot, CommandShaper, the rover model fitted to field bags -- with two
things swapped for indoors:

- Localization: DeadReckoning below instead of controller_sim's EKF+GPS model.
  That one has a bounded error (AR(1), 0.2 m); integrated wheels + gyro drift
  without bound. Errors swept, none measured on this rover yet:
    wheel_scale   odometry distance error. ekf_local's own measurement on
                  wuhan4_linear: +1.7% with use_control, -1.2% without.
    gyro_scale    yaw-rate scale error, i.e. degrees lost per degree turned.
    gyro_bias     deg/min of yaw drift while driving straight.
- World: the corridor walls of config/indoor_nyu_track.yaml, widths measured
  on the NYU map (median south 4.0 m, west 1.9, north 2.0, east 1.9), and five
  discs (likely tables) along the south corridor.

What it does NOT model, so read the results as the carrot's share alone: the
model's corridor following. The ModelEmulator only reproduces the outdoor
underturning (alpha = 0.117 x carrot bearing); it does not see walls. So a
"hit" is where the plan would have to rely on the model -- or on a correction
this stack does not have yet (cones, corners; docs/PLAN_INDOOR_NYU.md).

    source /opt/ros/jazzy/setup.bash && source ~/lsdc_ws/install/setup.bash
    python3 test/indoor_sim.py            # needs utm, PyYAML
"""

import argparse
import itertools
import math
import os
import re
import sys
from collections import deque
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import controller_sim as cs  # noqa: E402
import obstacles as obs  # noqa: E402
from erc_inference import indoor_track as it  # noqa: E402

TRACK_FILE = os.path.join(os.path.dirname(HERE), 'config', 'indoor_nyu_track.yaml')
INDOOR_NODE = os.path.join(os.path.dirname(HERE), 'erc_inference', 'indoor_mission_node.py')

# Corridors: (centerline, half width). Widths measured on the map image as the
# floor run across each corridor, 23 profiles per side (overlays masked):
# median west 1.9 m, east 1.9 m, north 2.0 m, south 4.0 m (p10 3.1 m, the half
# width used here is below even that).
SOUTH_Y, SOUTH_HW = -0.2, 1.55
EAST_X, EAST_HW = 36.2, 0.95
NORTH_Y, NORTH_HW = 27.8, 1.0
WEST_X, WEST_HW = 9.1, 0.95
START_END_X = -3.0
# Two corners are T-junctions on the map, not L's: the south corridor's floor
# carries on to x = 41.6 (~4.5 m past the east corridor), the north one to
# x = 5.2 (~3 m past the west corridor). Overshooting there is open floor -- and
# a wrong turn.
SOUTH_EAST_END_X = 41.6
NORTH_WEST_END_X = 5.2
# Five white discs ~0.5-0.65 m across along the south corridor's outer wall
# between CP1 and the SE corner, ringed by red points: round tables with chairs
# is the likely reading, not confirmed. Modelled as 0.35 m-radius discs.
SOUTH_DISCS = [(27.8, -1.3), (30.0, -1.3), (31.7, -1.3), (33.8, -1.4), (35.9, -1.4)]
DISC_R = 0.35


# The south corridor as measured on the map (28-sept, floor run every 0.5 m, x 11-34): the OUTER
# wall is straight at y ~ -2.35 (p10 -2.52, p90 -1.61 where the tables' points break the floor);
# the INNER (courtyard) wall is curved, y 1.2-2.9.
SOUTH_OUTER_Y = -2.35
SOUTH_INNER = [(10.05, 2.6), (14.0, 2.3), (18.0, 1.5), (22.0, 2.0), (27.0, 2.2), (31.0, 1.4), (35.25, 1.3)]


def nyu_walls():
    S = obs.Segment
    s_lo, s_hi = SOUTH_Y - SOUTH_HW, SOUTH_Y + SOUTH_HW
    e_lo, e_hi = EAST_X - EAST_HW, EAST_X + EAST_HW
    n_lo, n_hi = NORTH_Y - NORTH_HW, NORTH_Y + NORTH_HW
    w_lo, w_hi = WEST_X - WEST_HW, WEST_X + WEST_HW
    se, nw = SOUTH_EAST_END_X, NORTH_WEST_END_X
    s_lo = SOUTH_OUTER_Y
    outer = [S((START_END_X, s_lo), (se, s_lo)), S((se, s_lo), (se, s_hi)), S((se, s_hi), (e_hi, s_hi)),
             S((e_hi, s_hi), (e_hi, n_hi)), S((e_hi, n_hi), (nw, n_hi)), S((nw, n_hi), (nw, n_lo)),
             S((nw, n_lo), (w_lo, n_lo)), S((w_lo, n_lo), (w_lo, s_hi)),
             S((w_lo, s_hi), (START_END_X, s_hi)), S((START_END_X, s_hi), (START_END_X, s_lo))]
    si = SOUTH_INNER
    inner = [S(si[i], si[i + 1]) for i in range(len(si) - 1)] + [
             S((e_lo, si[-1][1]), (e_lo, n_lo)), S((e_lo, n_lo), (w_hi, n_lo)), S((w_hi, n_lo), (w_hi, si[0][1]))]
    return outer + inner + [obs.Disc(c, DISC_R) for c in SOUTH_DISCS]


def nyu_scenario():
    track = it.load_track(TRACK_FILE)
    legs = [list(c.via) + [(c.x, c.y)] for c in track.checkpoints]
    return cs.Scenario('nyu', (track.start.x, track.start.y, track.start.yaw), legs, nyu_walls())


class DeadReckoning:
    """Wheel speed + gyro integrated from the true start, as ekf_local does, with
    scale and bias errors. Same interface as controller_sim.Localization."""

    wheel_scale = 0.0
    gyro_scale = 0.0
    gyro_bias_deg_min = 0.0

    def __init__(self, model, rng):
        self.m = model
        self.rng = rng
        self.x = self.y = self.theta = None
        self._history = deque()

    def update(self, t, dt, rover):
        if self.x is None:
            self.x, self.y, self.theta = rover.x, rover.y, rover.theta
        w = rover.w * (1.0 + self.gyro_scale) + math.radians(self.gyro_bias_deg_min) / 60.0
        self.theta = cs.mc.clip_angle(self.theta + w * dt)
        v = rover.v * (1.0 + self.wheel_scale)
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self._history.append((t, self.x, self.y, self.theta))
        while len(self._history) > 1 and self._history[1][0] <= t - self.m.sensor_latency_s:
            self._history.popleft()

    def estimate(self):
        _, x, y, theta = self._history[0]
        return x, y, theta


def indoor_node_defaults():
    """checkpoint_controller_node's defaults, overridden by the numeric ones
    indoor_mission_node declares (checkpoint_proximity_m 2.0, ...)."""
    cp = cs.checkpoint_node_defaults()
    for name, value in re.findall(r"declare_parameter\('(\w+)', ([-0-9.]+)\)", open(INDOOR_NODE).read()):
        cp[name] = float(value)
    return cp


def run(wheel_scale, gyro_scale, gyro_bias, k_w_moving, seed, params=None, record=False):
    DeadReckoning.wheel_scale = wheel_scale
    DeadReckoning.gyro_scale = gyro_scale
    DeadReckoning.gyro_bias_deg_min = gyro_bias
    saved, cs.Localization = cs.Localization, DeadReckoning
    try:
        model = replace(cs.RoverModel(), k_w_moving=k_w_moving, heading_bias_deg=0.0)
        return cs.simulate(nyu_scenario(), params or cs.controller_params(), model, seed,
                           record=record, node_defaults=indoor_node_defaults())
    finally:
        cs.Localization = saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=2)
    args = ap.parse_args()
    grid = {
        'wheel_scale': [-0.02, 0.0, 0.02],
        'gyro_scale': [-0.03, 0.0, 0.03],
        'gyro_bias': [0.0, 1.0, 3.0],
        'k_w_moving': [0.36, 1.1],
    }
    rows = []
    for combo in itertools.product(*grid.values()):
        for seed in range(args.seeds):
            r = run(*combo, seed)
            rows.append((combo, seed, r))
    print(f"{'wheel':>6} {'gyro_sc':>7} {'bias':>5} {'k_w':>5} {'seed':>4} | {'done':>4} {'hit':>4} "
          f"{'clear_min':>9} {'arr_err':>7} {'xt_p95':>6} {'time':>6}")
    for (ws, gs, gb, kw), seed, r in rows:
        print(f"{ws:+6.2f} {gs:+7.2f} {gb:5.1f} {kw:5.2f} {seed:4d} | {str(r.completed)[0]:>4} "
              f"{str(r.hit)[0]:>4} {r.min_clearance_m:9.2f} {r.arrival_error_max:7.2f} "
              f"{r.cross_track_p95:6.2f} {r.time_s:6.0f}")
    print()
    for key, values in grid.items():
        idx = list(grid).index(key)
        for v in values:
            sel = [r for c, _, r in rows if c[idx] == v]
            print(f"{key}={v:+.2f}: completed {sum(r.completed for r in sel)}/{len(sel)}, "
                  f"hit a wall {sum(r.hit for r in sel)}/{len(sel)}, "
                  f"arrival error max p50 {np.median([r.arrival_error_max for r in sel]):.2f} m")
    clean = [r for c, _, r in rows if c[0] == 0.0 and c[1] == 0.0 and c[2] == 0.0]
    print(f"\nno drift at all: completed {sum(r.completed for r in clean)}/{len(clean)}, "
          f"hit {sum(r.hit for r in clean)}/{len(clean)}, "
          f"min clearance {min(r.min_clearance_m for r in clean):.2f} m")


if __name__ == '__main__':
    main()
