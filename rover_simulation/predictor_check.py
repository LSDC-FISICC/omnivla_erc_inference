#!/usr/bin/env python3
"""Does a fuller Smith predictor help the carrot (and so the two OmniVLA nodes,
which steer through the same MotionController)?

The controller already has one, reduced to yaw (motion_control.DelayCompensator):
it subtracts gain x the yaw-rate commands of the last 1.3 s from the carrot
bearing, with gain 1.0. Two things it leaves out:
  * position: in 1.3 s at 0.3 m/s the rover moves ~0.4 m, against a carrot 1.5 m ahead
  * the measured gain: ~0.36 of the commanded yaw rate while moving (1.18 in place)

Variants, all on the same scenarios, plants and seeds:
  none        delay_compensation_gain 0
  yaw_1.0     what runs today
  yaw_0.36    the yaw-only predictor with the measured moving gain
  full_pose_k1  the same with the moving yaw gain over-stated as 1.0, as yaw_1.0 does
  full_pose   position + yaw rolled forward through the last 1.3 s of commands with
              the measured gains (what nav2_route_follower_node does for MPPI), and
              the yaw-only one off -- every consumer of the pose sees the prediction

    source /opt/ros/jazzy/setup.bash && source ~/lsdc_ws/install/setup.bash
    ~/lsdc/erc-omni-vla/.venv/bin/python3 predictor_check.py [--seeds 2]
"""
import argparse
import itertools
import math
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import common

cs = common.cs
SCENARIOS = ('straight', 'L-turn', 'U-turn', 'zigzag', 'mission_16sept')
# the controller_sim clock-offset sweep, reduced: delay and the moving yaw gain
# are what a predictor depends on
PLANTS = [dict(delay_s=d, k_w_moving=k) for d, k in itertools.product((0.9, 1.3, 1.8), (0.2, 0.36, 1.0))]
PRED_DELAY, K_V, K_W_MOVING, K_W_IN_PLACE = 1.3, 1.11, 0.36, 1.18
# weave_check.py's field-like plant: reproduces mission_wuhan_hard's weave (5.7/min, 30.9 deg
# against 5.4/min, 31 deg), which the default plant does not (3.6/min, 12.2 deg)
FIELD = dict(heading_extra_latency_s=1.0, heading_hold_s=0.5, k_w_moving_sigma=1.6, position_sigma_m=0.5)


def predict(x, y, th, history, now, k_w_moving=K_W_MOVING):
    """Roll the estimated pose forward through the commands of the last PRED_DELAY."""
    t0 = now - PRED_DELAY
    seg = [c for c in history if c[0] > t0]
    before = [c for c in history if c[0] <= t0]
    if before:
        seg.insert(0, (t0, before[-1][1], before[-1][2]))
    seg.append((now, 0.0, 0.0))
    for (ta, v, w), (tb, _a, _b) in zip(seg[:-1], seg[1:]):
        vr = K_V * v if abs(v) >= 0.15 else 0.0
        wr = (k_w_moving if vr else K_W_IN_PLACE) * w
        n = max(1, int((tb - ta) / 0.05))
        h = (tb - ta) / n
        for _ in range(n):
            x += vr * math.cos(th) * h
            y += vr * math.sin(th) * h
            th += wr * h
    return x, y, th


def run(job):
    variant, scenario, plant, seed, field = job
    overrides = {'polar.steering_source': 'carrot'}
    if variant == 'none':
        overrides['polar.delay_compensation_gain'] = 0.0
    elif variant == 'yaw_0.36':
        overrides['polar.delay_compensation_gain'] = K_W_MOVING
    elif variant.startswith('full_pose'):
        overrides['polar.delay_compensation_gain'] = 0.0
    params = cs.controller_params(overrides)
    model = replace(cs.RoverModel(), **plant, **(FIELD if field else {}))
    node = cs.load_checkpoint_node()
    if variant.startswith('full_pose'):
        kwm = 1.0 if variant == 'full_pose_k1' else K_W_MOVING
        history = []
        clock = {'t': 0.0}
        real_command = cs.Rover.command
        real_estimate = cs.Localization.estimate
        real_update = cs.Localization.update

        def command(self, t, linear, angular):
            history.append((t, linear, angular))
            del history[:-400]
            return real_command(self, t, linear, angular)

        def update(self, t, dt, rover):
            clock['t'] = t
            return real_update(self, t, dt, rover)

        def estimate(self):
            x, y, th = real_estimate(self)
            return predict(x, y, th, history, clock['t'], kwm)
        cs.Rover.command, cs.Localization.update, cs.Localization.estimate = command, update, estimate
    r = cs.simulate(cs.SCENARIOS[scenario], params, model, seed=seed, node=node)
    return variant, scenario, r.completed, r.weave_per_min, r.cross_track_p95, r.time_s, r.stalled_s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--field', action='store_true', help='add the field-like effects to every plant')
    a = ap.parse_args()
    variants = ('none', 'yaw_1.0', 'yaw_0.36', 'full_pose', 'full_pose_k1')
    jobs = [(v, s, p, i, a.field) for v in variants for s in SCENARIOS for p in PLANTS for i in range(a.seeds)]
    # one process per job: full_pose patches controller_sim's classes in place
    with Pool(12, maxtasksperchild=1) as pool:
        res = pool.map(run, jobs, chunksize=1)
    print(f'{len(SCENARIOS)} scenarios x {len(PLANTS)} plants (delay 0.9/1.3/1.8 s x moving yaw gain '
          f'0.2/0.36/1.0) x {a.seeds} seeds' + ('  + FIELD-LIKE effects' if a.field else ''))
    print(f"{'variant':10s} {'done':>7} {'weave/min p50':>14} {'p95':>6} {'cross-track p95 (m) p50':>24} "
          f"{'p95':>6} {'time p50':>9}")
    for v in variants:
        R = [r for r in res if r[0] == v]
        w = np.array([r[3] for r in R]); c = np.array([r[4] for r in R]); t = np.array([r[5] for r in R])
        print(f'{v:10s} {sum(r[2] for r in R):3d}/{len(R):<3d} {np.median(w):14.1f} {np.percentile(w, 95):6.1f} '
              f'{np.median(c):24.2f} {np.percentile(c, 95):6.2f} {np.median(t):8.0f}s')


if __name__ == '__main__':
    main()
