"""The outdoor GPS mission with carrot_controller_node's drive_mode reactive, against today's polar
drive, in controller_sim's obstacle worlds -- on the team's own Mini+ plant and on the competition one.

    PY=~/lsdc/erc-omni-vla/.venv/bin/python3
    $PY reactive_check.py --seeds 6                       # table
    $PY reactive_check.py --seeds 6 --plants mini --fails

Configurations (what the launch would run):
    polar          polar law + obstacle_sidestep + local_replan + recovery.explore: the
                   recommended outdoor setting of 28-sept (84/84, 0 hits on the competition plant)
    reactive       ReactiveDrive on the carrot's bearing, no local map
    reactive+lp    + local_replan + recovery.explore (the A* detour gives the carrot)
    reactive+lp-corr  the same with the corridor rule (local.unseen_extend_m 0.1)
    reactive-c4    reactive alone with the carrot 4 m ahead (instead of replanning, aim further
                   along the route)
The reactive drive is the robot's own code (erc_controller/reactive_drive.py) through the same
inputs carrot_controller_node gives it; the profile here is instantaneous (no 0.5 s image lag).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import cs, obs  # noqa: E402
from erc_controller.reactive_drive import ReactiveDrive  # noqa: E402

SCENARIOS = 'hedge,planter,wall,kerb,chicane,post,straight,zigzag,mission_16sept'
# recommended 28-sept: rejoin with A*'s own unseen rule, explore when there is no path
LP = dict(rejoin_respects_unseen=True, unseen_extend_m=2.5)
# corridor rule (loop_patrol.LOCAL_INDOOR): in a 2 m walkway every unseen cell is near a wall; with 2.5 m all of
# them are lethal and A* answers "no path" (Galileo plaza, 29-oct)
LP_CORRIDOR = dict(rejoin_respects_unseen=True, unseen_extend_m=0.1)
REC = dict(explore=True, look=False)

PLANTS = {
    # config/rover_dynamics.yaml (12-oct): dead time 0.65 s, tau 0.74 / 0.62 s, k_v 1.07, moves from
    # 0.10 m/s; moving k_w = 1.03 + 0.40 v (~1.1 at 0.2 m/s); in place 1.7 (|w| - 0.10) ~ 1.13 at 0.3
    'mini': dict(delay_s=0.65, tau_v=0.74, tau_w=0.62, k_v=1.07, k_w_moving=1.11, k_w_in_place=1.13,
                 v_breakaway=0.08, w_breakaway=0.12),
    'competition': {},
}


class ReactiveOverride:
    """controller_sim's override hook -> ReactiveDrive, as carrot_controller_node feeds it."""
    drives = True

    def __init__(self, **kw):
        self.d = ReactiveDrive(**kw)
        self.state = 'reactive'

    def step(self, t, yaw, cur, bearings, free, v, w, bearing):
        caps = obs._range_cap(np.asarray(bearings, float))
        hit = np.asarray(free, float) < caps - 1e-3
        v, w = self.d.step(t, yaw, (bearings, free, hit), yaw, np.degrees(bearing))
        return v, w, '', False


def smoothness(res):
    """(share of the mission time the rover really moves, stops per minute) from the recorded truth."""
    v = np.abs(res.trace['true'][:, 0])
    moving = v > 0.05
    stops = int(np.sum(moving[:-1] & ~moving[1:]))
    minutes = max(res.time_s, 1.0) / 60.0
    return float(moving.mean()), stops / minutes


def run(job):
    config, plant, scenario, seed = job
    model = cs.RoverModel(**PLANTS[plant])
    if config == 'polar':
        res, _ = common.run(scenario, seed, 'lp+ss', 1.5, lp=LP, model=model, recovery=REC, record=True)
    else:
        override = ReactiveOverride()
        lp = LP_CORRIDOR if config.endswith('-corr') else LP
        res, _ = common.run(scenario, seed, 'lp' if 'lp' in config else 'carrot',
                            4.0 if config.endswith('c4') else 1.5,
                            lp=lp, model=model, override=override,
                            recovery=REC if 'lp' in config else None, record=True)
    res.moving, res.stops_min = smoothness(res)
    res.trace = None                                  # not picklable back through the pool, not needed
    return config, plant, scenario, seed, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--scenarios', default=SCENARIOS)
    ap.add_argument('--configs', default='polar,reactive,reactive-c4,reactive+lp')
    ap.add_argument('--plants', default='mini,competition')
    ap.add_argument('--processes', type=int, default=16)
    ap.add_argument('--fails', action='store_true')
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    jobs = [(c, p, s, i) for c in a.configs.split(',') for p in a.plants.split(',')
            for s in a.scenarios.split(',') for i in range(a.seeds)]
    with mp.Pool(a.processes) as pool:
        out = pool.map(run, jobs)
    rows = []
    print(f"{'config':12s} {'plant':12s} {'done':>7s} {'hits':>5s} {'t_p50':>6s} {'xtrack_p95':>10s} "
          f"{'clear_min':>9s} {'stalled':>7s} {'replans':>7s} {'moving':>6s} {'stops/min':>9s}")
    for c in a.configs.split(','):
        for p in a.plants.split(','):
            rs = [r for (cc, pp, _s, _i, r) in out if cc == c and pp == p]
            done = sum(r.completed for r in rs)
            hits = sum(r.hit for r in rs)
            t50 = np.median([r.time_s for r in rs if r.completed]) if done else float('nan')
            x95 = np.nanmedian([r.cross_track_p95 for r in rs])
            cmin = np.min([r.min_clearance_m for r in rs])
            st = np.median([r.stalled_s for r in rs])
            rp = sum(r.replans for r in rs)
            mv = np.median([r.moving for r in rs])
            sm = np.median([r.stops_min for r in rs])
            print(f'{c:12s} {p:12s} {done:3d}/{len(rs):<3d} {hits:5d} {t50:6.0f} {x95:10.2f} {cmin:9.2f} '
                  f'{st:7.1f} {rp:7d} {100 * mv:5.0f}% {sm:9.1f}')
            rows.append(dict(config=c, plant=p, done=done, n=len(rs), hits=hits, t_p50=t50, xtrack_p95=x95,
                             clearance_min=cmin, stalled_p50=st, replans=rp, moving=mv, stops_per_min=sm))
    if a.fails:
        for c, p, s, i, r in out:
            if not r.completed or r.hit:
                print(f'  {c} {p} {s} seed {i}: completed={r.completed} hit={r.hit} '
                      f'clear={r.min_clearance_m:.2f} t={r.time_s:.0f}')
    if a.json:
        json.dump(dict(rows=rows, per_run=[dict(config=c, plant=p, scenario=s, seed=i, completed=r.completed,
                                                hit=r.hit, time_s=r.time_s, clearance=r.min_clearance_m)
                                           for c, p, s, i, r in out]),
                  open(a.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
