#!/usr/bin/env python3
"""Can the simulator reproduce the field weave of mission_wuhan_hard?

Field (docs/MissionWuhanHard.md, carrot 1.5 m): yaw-rate sign changes 5.4/min,
median |bearing to the carrot| 31 deg, period ~20 s. The default simulator gives
3.0/min and 12.6 deg: same period, a third of the amplitude.

Three field effects the default plant leaves out, each switchable in RoverModel:
  heading_extra_latency_s  the compass arrives older than the EKF position (extrapolated
                           to now). Telemetry is ~0.25-0.3 s in transit (data_latency),
                           plus the heading node's hold; 0.5-1.0 s tried
  heading_hold_s           telemetry comes in bursts every ~0.5 s: the heading steps
  k_w_moving_sigma         the moving yaw gain varies a lot (IQR 0.16-1.42): log-normal
                           multiplier redrawn every 2 s
  position_sigma_m         (already in RoverModel, 0.2 m by default): with the carrot 1.5 m
                           ahead, 1 m of position error is ~30 deg of bearing

    source /opt/ros/jazzy/setup.bash && source ~/lsdc_ws/install/setup.bash
    ~/lsdc/erc-omni-vla/.venv/bin/python3 weave_check.py [--seeds 4]
"""
import argparse
import math
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import common

cs = common.cs
SCENARIOS = ('straight', 'zigzag', 'mission_16sept')
VARIANTS = {
    'default': {},
    'heading +0.5 s': dict(heading_extra_latency_s=0.5),
    'heading +1.0 s': dict(heading_extra_latency_s=1.0),
    'heading hold 0.5 s': dict(heading_hold_s=0.5),
    'k_w sigma 0.8': dict(k_w_moving_sigma=0.8),
    'k_w sigma 1.6': dict(k_w_moving_sigma=1.6),
    'all: +0.5, hold, sigma 0.8': dict(heading_extra_latency_s=0.5, heading_hold_s=0.5, k_w_moving_sigma=0.8),
    'all: +1.0, hold, sigma 1.6': dict(heading_extra_latency_s=1.0, heading_hold_s=0.5, k_w_moving_sigma=1.6),
    'position error 0.5 m': dict(position_sigma_m=0.5),
    'position error 1.0 m': dict(position_sigma_m=1.0),
    'all + position 0.5 m': dict(heading_extra_latency_s=1.0, heading_hold_s=0.5, k_w_moving_sigma=1.6,
                                 position_sigma_m=0.5),
    'all + position 1.0 m': dict(heading_extra_latency_s=1.0, heading_hold_s=0.5, k_w_moving_sigma=1.6,
                                 position_sigma_m=1.0),
}


def run(job):
    name, scenario, seed, carrot = job
    node = cs.load_checkpoint_node()
    d = dict(cs.checkpoint_node_defaults())
    d['carrot_distance_m'] = carrot
    params = cs.controller_params({'polar.steering_source': 'carrot'})
    model = replace(cs.RoverModel(), **VARIANTS[name])
    r = cs.simulate(cs.SCENARIOS[scenario], params, model, seed=seed, node=node, node_defaults=d, record=True)
    pol = [tk for tk in r.trace['ticks'] if tk[2] == 'polar']
    b = [abs(math.degrees(tk[3])) for tk in pol]
    # the field's 5.4/min counts sign changes of the COMMANDED omega in /omnivla_debug
    w = np.array([tk[6] for tk in pol])
    sg = np.sign(w[np.abs(w) > 1e-3])
    minutes = (pol[-1][0] - pol[0][0]) / 60.0 if len(pol) > 1 else 1.0
    cmd_changes = float(np.sum(sg[1:] != sg[:-1]) / minutes) if len(sg) > 1 else 0.0
    return (name, scenario, r.completed, r.weave_per_min, float(np.median(b)) if b else float('nan'),
            r.cross_track_p95, cmd_changes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--carrot', type=float, default=1.5)
    ap.add_argument('--only', default='', help='comma list of variant names')
    a = ap.parse_args()
    names = [v for v in VARIANTS if not a.only or v in a.only.split(',')]
    jobs = [(v, s, i, a.carrot) for v in names for s in SCENARIOS for i in range(a.seeds)]
    with Pool(12) as pool:
        res = pool.map(run, jobs, chunksize=1)
    print(f'carrot {a.carrot} m, {len(SCENARIOS)} scenarios x {a.seeds} seeds.   FIELD (wuhan_hard): 5.4/min, 31 deg')
    print(f"{'variant':30s} {'done':>6} {'cmd sign/min':>13} {'true w sign/min':>16} {'median |bearing| deg':>21} {'cross-track p95':>16}")
    for v in names:
        R = [r for r in res if r[0] == v]
        print(f'{v:30s} {sum(r[2] for r in R):3d}/{len(R):<2d} {np.median([r[6] for r in R]):13.1f} '
              f'{np.median([r[3] for r in R]):16.1f} '
              f'{np.median([r[4] for r in R]):21.1f} {np.median([r[5] for r in R]):15.2f}m')


if __name__ == '__main__':
    main()
