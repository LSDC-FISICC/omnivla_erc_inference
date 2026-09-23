#!/usr/bin/env python3
"""Configs x scenarios x seeds, in parallel. The table the MissionCarrotSidestep numbers came from.

    python3 sweep.py --carrot 1.5 --seeds 8 --configs ss,lp+ss
    python3 sweep.py --scenarios hedge,wall --lp '{"inflate_m": 0.45}' --ss '{"stop_distance_m": 1.0}'
    python3 sweep.py --fails          # also print every run that hit or did not finish
"""
import argparse
import collections
import json
from multiprocessing import Pool

import numpy as np

import common

OBSTACLE_SCENARIOS = 'hedge,planter,wall,kerb,chicane,post,straight,mission_16sept'


def _job(args):
    config, scenario, seed, carrot, ss, lp = args
    r, side = common.run(scenario, seed, config, carrot, ss, lp)
    return dict(config=config, scenario=scenario, seed=seed, hit=r.hit, done=r.completed,
                time=r.time_s, clear=r.min_clearance_m, replans=r.replans,
                final=side.state if side else '-', notes=r.trace['local_notes'][:3])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--carrot', type=float, default=1.5, help='carrot_distance_m')
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--scenarios', default=OBSTACLE_SCENARIOS)
    ap.add_argument('--configs', default='ss,lp+ss', help=f'comma list of {common.CONFIGS}')
    ap.add_argument('--ss', default='{}', help='SideStep overrides, JSON')
    ap.add_argument('--lp', default='{}', help='LocalReplanner overrides, JSON')
    ap.add_argument('--processes', type=int, default=12)
    ap.add_argument('--fails', action='store_true')
    a = ap.parse_args()
    ss, lp = json.loads(a.ss), json.loads(a.lp)
    configs, scenarios = a.configs.split(','), a.scenarios.split(',')
    jobs = [(c, s, i, a.carrot, ss, lp) for c in configs for s in scenarios for i in range(a.seeds)]
    with Pool(a.processes) as pool:
        res = pool.map(_job, jobs)
    print(f'carrot {a.carrot} m, {a.seeds} seeds, ss={ss} lp={lp}')
    print(f"{'config':7s} {'scenario':15s} {'hit':>5} {'done':>6} {'time':>6} {'min clear':>9} {'replans':>7}  side-step end states")
    for c in configs:
        for s in scenarios:
            rs = [r for r in res if r['config'] == c and r['scenario'] == s]
            n = len(rs)
            print(f"{c:7s} {s:15s} {sum(r['hit'] for r in rs):2d}/{n:<2d} {sum(r['done'] for r in rs):3d}/{n:<2d} "
                  f"{np.mean([r['time'] for r in rs]):5.0f}s {min(r['clear'] for r in rs):8.2f}m "
                  f"{np.mean([r['replans'] for r in rs]):7.1f}  {dict(collections.Counter(r['final'] for r in rs))}")
    if a.fails:
        for r in res:
            if r['hit'] or not r['done']:
                print('FAIL', {k: r[k] for k in ('config', 'scenario', 'seed', 'hit', 'done', 'final')}, r['notes'])


if __name__ == '__main__':
    main()
