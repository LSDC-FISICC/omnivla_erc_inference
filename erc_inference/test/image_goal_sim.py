#!/usr/bin/env python3
"""Image-goal missions on the NYU indoor loop with the cones placed at random, in closed loop.

The mission is the PDF's list of images (CP1 red, CP2 blue, CP3 green, CP4
yellow, Finish orange); the cones' positions change on the day and there is
no map. Runs image_goal_mission.ImageGoalMission (scan, explore, approach,
LocalReplanner + BlockedRecovery) with MotionController (carrot steering, as
carrot_controller_node) and CommandShaper, optionally SafetyEnvelope's
obstacle stop and the SideStep reflex, on controller_sim's plant, indoor_sim's
dead reckoning and the NYU walls + tables (+ cone_sim's clutter). Cone
detection and image lag are emulated as in cone_sim.py.

Cone layouts:
  loop    CP1..CP4 in order along the loop, direction of travel unknown to the rover
  random  anywhere on the loop, any order
The orange Start/Finish cone stands 0.8 m behind the rover at the start.

Judged on the truth: the SDK is emulated as accepting an arrival within
judge_m of the true cone (rejections make the rover close in).

    python3 test/image_goal_sim.py [--seeds 4]
"""
import argparse
import itertools
import json
import math
import os
import sys
from collections import deque
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import controller_sim as cs  # noqa: E402
import cone_sim  # noqa: E402
import indoor_sim as isim  # noqa: E402
import obstacles as obs  # noqa: E402
from erc_inference import motion_control as mc  # noqa: E402
from erc_inference.image_goal_mission import ARRIVED, Goal, ImageGoalMission  # noqa: E402
from erc_inference.safety_envelope import DEFAULTS as SAFETY, SafetyEnvelope  # noqa: E402
from erc_inference.sidestep import SideStep  # noqa: E402

JUDGE_M = 2.0
CONE_R = cone_sim.CONE_R
MISSION = [('CP1', 'red_orange', 'red'), ('CP2', 'blue', 'blue'), ('CP3', 'green', 'green'),
           ('CP4', 'yellow', 'yellow'), ('Finish', 'red_orange', 'orange')]

# the loop's centerline, CCW from the start (track frame of indoor_nyu_track.yaml)
LOOP = np.array([(0.0, -0.2), (36.2, -0.2), (36.2, 27.8), (9.1, 27.8), (9.1, -0.2)])


def loop_point(s):
    """Point at arc length s along LOOP (s from the start, wrapping over the ring part)."""
    pts = LOOP
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(min(np.searchsorted(cum, s, side='right') - 1, len(pts) - 2))
    t = (s - cum[i]) / (cum[i + 1] - cum[i])
    return pts[i] + t * (pts[i + 1] - pts[i]), (pts[i + 1] - pts[i]) / (cum[i + 1] - cum[i])


def place_cones(layout, rng, walls):
    """True positions of CP1..CP4 (the Finish is the start cone)."""
    total = float(np.sum(np.hypot(*np.diff(LOOP, axis=0).T)))
    for _ in range(200):
        if layout == 'loop':
            ss = np.sort(rng.uniform(12.0, total - 6.0, 4))
            if rng.random() < 0.5:                      # the other direction of travel
                ss = ss[::-1]
        else:
            ss = rng.uniform(8.0, total - 4.0, 4)
        pos = []
        for s in ss:
            p, d = loop_point(s)
            n = np.array([-d[1], d[0]])
            pos.append(p + n * rng.uniform(-0.3, 0.3))
        far = all(np.hypot(*(a - b)) > 4.0 for a, b in itertools.combinations(pos, 2))
        clear = all(min(w.distance_to(p) for w in walls) > 0.4 for p in pos)
        if far and clear:
            return pos
    raise RuntimeError('could not place cones')


def run(layout='loop', scenario='clutter', seed=0, drift=(0.01, 0.01, 0.5), k_w=1.1, image_lag_s=0.8,
        stop=True, ss=False, gt45=True, lp_params=None, rec_params=None, mission_params=None,
        t_limit=2400.0, leg_limit_s=700.0, stop_distance_m=0.7, max_linear=0.25, record=False):
    rng = np.random.default_rng(seed)
    walls = isim.nyu_walls() + cone_sim.SCENARIOS[scenario]
    cps = place_cones(layout, rng, walls)
    start_cone = np.array([-0.8, 0.0])
    cone_xy = {'CP1': cps[0], 'CP2': cps[1], 'CP3': cps[2], 'CP4': cps[3], 'Finish': start_cone}
    cone_cls = {n: c for n, c, _ in MISSION}
    cone_objs = {n: obs.Disc(p, CONE_R) for n, p in cone_xy.items()}
    world_all = walls + list(cone_objs.values())

    overrides = {'polar.steering_source': 'carrot'}
    if gt45:
        overrides.update({'goal_turn.enter_deg': 45.0, 'goal_turn.exit_deg': 15.0})
    params = cs.controller_params(overrides)
    model = replace(cs.RoverModel(), k_w_moving=k_w, heading_bias_deg=0.0)
    cone_sim.Drift.wheel_scale, cone_sim.Drift.gyro_scale, cone_sim.Drift.gyro_bias_deg_min = drift
    rover = cs.Rover(model, 0.0, 0.0, 0.0, rng=np.random.default_rng(seed + 1000))
    loc = cone_sim.Drift(model, rng)
    controller = mc.MotionController(params)
    shaper = mc.CommandShaper(0.3, 0.6, 0.6, 1.2)
    envelope = SafetyEnvelope() if stop else None
    safety = dict(SAFETY, **{'safety.tilt_enabled': False, 'safety.obstacle_enabled': True,
                             'safety.stop_distance_m': stop_distance_m, 'safety.max_linear_vel': max_linear})
    sidestep = SideStep() if ss else None
    goals = [Goal(n, c, i + 1, is_start_cone=(n == 'Finish')) for i, (n, c, _) in enumerate(MISSION)]
    mission = ImageGoalMission(goals, (0.0, 0.0), lp_params, rec_params, **(mission_params or {}))

    straight = [(0.4 * (i + 1), 0.0, 1.0, 0.0) for i in range(8)]
    tick, out_period, cam_period, latency = 1.0 / params['tick_rate'], 0.1, 1.0 / 3.0, 0.17
    history = deque()
    yaws = deque()
    results, hits, hit_now = [], [], False
    next_tick = next_out = next_cam = 0.0
    pending, model_cmd = None, (0.0, 0.0)
    step = None
    confirm_at = None
    leg_t0 = 0.0
    min_clear = math.inf
    trace = []
    t, dt = 0.0, 0.02
    loc.update(t, dt, rover)
    while t < t_limit and not mission.done:
        loc.update(t, dt, rover)
        ex, ey, eth = loc.estimate()
        history.append((t, rover.x, rover.y, rover.theta))
        while len(history) > 1 and history[1][0] <= t - image_lag_s:
            history.popleft()
        yaws.append((t, eth))
        while len(yaws) > 1 and yaws[0][0] < t - 1.0:
            yaws.popleft()
        # ---- camera: free space + cones, from the lagged true pose ----
        if t >= next_cam:
            next_cam += cam_period
            _, lx, ly, lth = history[0]
            pb, pf, pc = obs.profile(lx, ly, lth, world_all, rng)
            mission.observe_scan(t, ex, ey, eth, pb, pf, pf < pc - 1e-3)
            if envelope is not None:
                envelope.observe_free_space(t, pb, np.where(pf < pc - 1e-3, pf, 3.0), 3.0)
            o = np.array([lx, ly])
            for n, cpos in cone_xy.items():
                d = cpos - o
                r_true = float(np.hypot(*d))
                brg = math.degrees(cs.mc.clip_angle(math.atan2(d[1], d[0]) - lth))
                if abs(brg) > 55.0 or r_true > 10.0 or rng.random() > 0.9:
                    continue
                if not cone_sim.visible(o, cpos, walls + [c for k, c in cone_objs.items() if k != n], 10.0):
                    continue
                mission.observe_cone(ex, ey, eth, cone_cls[n], brg + rng.normal(0, 0.7),
                                     r_true * (1 + rng.normal(0, 0.03)))
        # ---- the mission node + the carrot controller, 3 Hz ----
        if t >= next_tick:
            next_tick += tick
            rate = 0.0
            if len(yaws) >= 2 and yaws[-1][0] - yaws[0][0] > 0.3:
                rate = abs(cs.mc.clip_angle(yaws[-1][1] - yaws[0][1])) / (yaws[-1][0] - yaws[0][0])
            if confirm_at is not None:
                cmd = (0.0, 0.0)
                if t >= confirm_at:
                    g = mission.goal
                    err = float(np.hypot(rover.x - cone_xy[g.name][0], rover.y - cone_xy[g.name][1]))
                    ok = err <= JUDGE_M
                    if ok or t - leg_t0 > leg_limit_s:
                        results.append(dict(name=g.name, reached=ok, err=round(err, 2), time=round(t - leg_t0, 1)))
                        leg_t0 = t
                        if not ok:           # give up on this one: count it and move on
                            mission.confirm(t, True)
                        else:
                            mission.confirm(t, True)
                    else:
                        mission.confirm(t, False)
                    confirm_at = None
                    controller.reset()
                model_cmd, pending = (0.0, 0.0), None
            else:
                step = mission.update(t, ex, ey, eth, rate)
                if step.arrived:
                    confirm_at = t + 1.5
                    model_cmd, pending = (0.0, 0.0), None
                elif t - leg_t0 > leg_limit_s:
                    g = mission.goal
                    err = float(np.hypot(rover.x - cone_xy[g.name][0], rover.y - cone_xy[g.name][1]))
                    results.append(dict(name=g.name, reached=False, err=round(err, 2), time=round(t - leg_t0, 1)))
                    leg_t0 = t
                    mission.confirm(t, True)
                    controller.reset()
                elif step.command is not None:
                    controller.note_idle(t)
                    pending = (t + latency, step.command)
                else:
                    cx, cy, _ = step.carrot
                    rel_x, rel_y = mc.robot_frame_offset(cx - ex, cy - ey, 90.0 - math.degrees(eth))
                    bearing = mc.goal_bearing_from_offset(rel_x, rel_y)
                    _m, vv, ww, _ = controller.command(t + latency, params, bearing, math.hypot(rel_x, rel_y),
                                                       straight, 4, True)
                    if sidestep is not None:
                        _, lx, ly, lth = history[0]
                        ob, of, _oc = obs.profile(lx, ly, lth, world_all, rng)
                        vv, ww, _n, resumed = sidestep.step(t, eth, (ex, ey), ob, of, vv, ww, bearing)
                        if resumed:
                            controller.reset()
                    if envelope is not None:
                        vv, ww, _h, _n = envelope.limit(t, vv, ww, safety)
                    pending = (t + latency, (vv, ww))
        if pending is not None and t >= pending[0]:
            model_cmd, pending = pending[1], None
        # ---- output, 10 Hz ----
        if t >= next_out:
            next_out += out_period
            lin, ang = shaper.step(model_cmd[0], model_cmd[1], out_period)
            rover.command(t, lin, ang)
        if record and (not trace or t - trace[-1][0] >= 0.5):
            trace.append((round(t, 1), mission.index, mission.state, round(rover.x, 2), round(rover.y, 2),
                          round(ex, 2), round(ey, 2), round(math.degrees(rover.theta)), model_cmd))
        rover.step(t, dt)
        c = obs.clearance((rover.x, rover.y), world_all)
        min_clear = min(min_clear, c)
        now_hit = c <= obs.FOOTPRINT_W / 2
        if now_hit and not hit_now:
            hits.append((round(t, 1), round(rover.x, 1), round(rover.y, 1)))
        hit_now = now_hit
        t += dt
    while len(results) < len(MISSION):
        g = mission.goal
        results.append(dict(name=g.name if g else '?', reached=False, err=float('nan'), time=float('nan')))
        mission.index += 1
    return dict(layout=layout, scenario=scenario, seed=seed, reached=sum(r['reached'] for r in results),
                legs=len(MISSION), hits=len(hits), hit_at=hits, min_clear=min_clear, time=round(t, 1),
                results=results, replans=mission.replanner.replans, looks=mission.recovery.looks,
                explores=mission.recovery.explores, notes=mission.notes, trace=trace,
                cones={k: tuple(np.round(v, 1)) for k, v in cone_xy.items()})


def _job(a):
    return run(**a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--layouts', default='loop,random')
    ap.add_argument('--scenarios', default='tables,clutter')
    ap.add_argument('--configs', default='stop,stop+ss,none')
    ap.add_argument('--lag', type=float, default=0.8)
    ap.add_argument('--kw', type=float, default=1.1)
    ap.add_argument('--mission', default='{}', help='ImageGoalMission overrides, JSON')
    ap.add_argument('--lp', default='{}')
    ap.add_argument('--rec', default='{}')
    args = ap.parse_args()
    jobs = []
    for layout, scen, cfg, seed in itertools.product(args.layouts.split(','), args.scenarios.split(','),
                                                     args.configs.split(','), range(args.seeds)):
        jobs.append(dict(layout=layout, scenario=scen, seed=seed, image_lag_s=args.lag, k_w=args.kw,
                         stop='stop' in cfg, ss='ss' in cfg, mission_params=json.loads(args.mission),
                         lp_params=json.loads(args.lp), rec_params=json.loads(args.rec)))
    with Pool() as pool:
        res = pool.map(_job, jobs)
    for j, r in zip(jobs, res):
        r['cfg'] = ('stop' if j['stop'] else '') + ('+ss' if j['ss'] else '') or 'none'
    print(f"{'layout':7s} {'scen':8s} {'cfg':8s} | {'cps ok':>7s} {'hits':>5s} {'w/hit':>6s} {'min_clr':>7s} "
          f"{'time p50':>8s} {'replan':>6s} {'looks':>5s} {'explr':>5s}")
    key = lambda r: (r['layout'], r['scenario'], r['cfg'])  # noqa: E731
    for k, grp in itertools.groupby(sorted(res, key=key), key=key):
        g = list(grp)
        print(f"{k[0]:7s} {k[1]:8s} {k[2]:8s} | {sum(r['reached'] for r in g):3d}/{sum(r['legs'] for r in g):<3d} "
              f"{sum(r['hits'] for r in g):5d} {sum(1 for r in g if r['hits']):3d}/{len(g):<2d} "
              f"{min(r['min_clear'] for r in g):7.2f} {np.median([r['time'] for r in g]):8.0f} "
              f"{np.mean([r['replans'] for r in g]):6.1f} {np.mean([r['looks'] for r in g]):5.1f} "
              f"{np.mean([r['explores'] for r in g]):5.1f}")


if __name__ == '__main__':
    main()
