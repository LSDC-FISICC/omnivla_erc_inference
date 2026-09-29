#!/usr/bin/env python3
"""NYU indoor, image-goal mission on the KNOWN loop: patrol, map cones, visit them in order.

Runs loop_patrol.LoopPatrolMission + WallLocalizer (with LocalReplanner and
BlockedRecovery inside), MotionController (carrot steering, as
carrot_controller_node), SafetyEnvelope's obstacle stop, CommandShaper, on
controller_sim's plant with indoor_sim's dead reckoning. World: the NYU walls
and tables, plus what may appear on the day:
  chairs  a chair in each narrow corridor, against a wall or near the middle
  doors   open doors: 1 m gaps in corridor walls, with a room behind (open floor)
Cones are placed at random along the loop, in the mission's order or not, and
the orange Start/Finish cone stands 0.8 m behind the rover. Cone detection and
image lag as in cone_sim.py. The SDK is emulated as accepting an arrival within
judge_m of the true cone.

    python3 test/loop_patrol_sim.py [--seeds 6]
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
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import controller_sim as cs  # noqa: E402
import cone_sim  # noqa: E402
import image_goal_sim as igs  # noqa: E402
import indoor_sim as isim  # noqa: E402
import obstacles as obs  # noqa: E402
from erc_inference import motion_control as mc  # noqa: E402
from erc_inference.image_goal_mission import Goal  # noqa: E402
from erc_inference.loop_patrol import LOCAL_INDOOR, SAFETY_INDOOR, KnownMap, LoopPatrolMission, WallLocalizer  # noqa: E402
from erc_inference.safety_envelope import DEFAULTS as SAFETY, SafetyEnvelope  # noqa: E402

JUDGE_M = 2.0
MISSION = igs.MISSION
TRACK = yaml.safe_load(open(isim.TRACK_FILE))


def chairs(rng):
    """One chair (r 0.25-0.3 m) in each narrow corridor, against a wall or near the middle."""
    out = []
    for cx, cy, axis_x in [(36.2, None, True), (None, 27.8, False), (9.1, None, True)]:
        along = rng.uniform(5.0, 23.0)
        off = rng.choice([-0.55, 0.55, rng.uniform(-0.3, 0.3)])
        if axis_x:
            out.append(obs.Disc((cx + off, along), rng.uniform(0.25, 0.3)))
        else:
            out.append(obs.Disc((9.1 + along, cy + off), rng.uniform(0.25, 0.3)))
    return out


def walls_with_doors(rng, n=3):
    """The NYU walls with n open doors (1 m gaps) in the narrow corridors' outer walls."""
    base = isim.nyu_walls()
    walls, tables = base[:10 + 4], base[14:]
    gaps = []
    for _ in range(n):
        which = rng.choice(['east', 'north', 'west'])
        along = rng.uniform(4.0, 24.0)
        gaps.append((which, along))
    out = []
    for w in walls:
        segs = [w]
        for which, along in gaps:
            nxt = []
            for sg in segs:
                a, b = sg.a, sg.b
                vertical = abs(a[0] - b[0]) < 1e-6
                if which == 'east' and vertical and abs(a[0] - (isim.EAST_X + isim.EAST_HW)) < 1e-6:
                    c = along
                elif which == 'west' and vertical and abs(a[0] - (isim.WEST_X - isim.WEST_HW)) < 1e-6:
                    c = along
                elif which == 'north' and not vertical and abs(a[1] - (isim.NORTH_Y + isim.NORTH_HW)) < 1e-6:
                    c = 9.1 + along
                else:
                    nxt.append(sg)
                    continue
                k = 1 if vertical else 0
                lo, hi = sorted([a[k], b[k]])
                if not (lo < c - 0.5 and c + 0.5 < hi):
                    nxt.append(sg)
                    continue
                p1, p2 = a.copy(), b.copy()
                if a[k] < b[k]:
                    m1, m2 = a.copy(), b.copy()
                    m1[k], m2[k] = c - 0.5, c + 0.5
                    nxt += [obs.Segment(p1, m1), obs.Segment(m2, p2)]
                else:
                    m1, m2 = a.copy(), b.copy()
                    m1[k], m2[k] = c + 0.5, c - 0.5
                    nxt += [obs.Segment(p1, m1), obs.Segment(m2, p2)]
            segs = nxt
        out += segs
    return out + tables, gaps


def run(layout='loop', world='chairs', seed=0, drift=(0.01, 0.01, 0.5), k_w=1.1, image_lag_s=0.8,
        localize=True, stop=False, stop_distance_m=0.7, max_linear=0.25, gt45=True, direction=1,
        mission_params=None, lp_params=None, rec_params=None, t_limit=2400.0, leg_limit_s=700.0,
        goals=None, stuck=True, record=False):
    """goals: override the mission (e.g. [Goal('X', 'never')] to patrol only)."""
    rng = np.random.default_rng(seed)
    if world in ('doors', 'chairs+doors'):
        walls, gaps = walls_with_doors(rng)
    else:
        walls, gaps = isim.nyu_walls(), []
    if 'chairs' in world:
        walls = walls + chairs(rng)
    cps = igs.place_cones(layout, rng, walls)
    cone_xy = {'CP1': cps[0], 'CP2': cps[1], 'CP3': cps[2], 'CP4': cps[3], 'Finish': np.array([-0.8, 0.0])}
    cone_cls = {n: c for n, c, _ in MISSION}
    cone_objs = {n: obs.Disc(p, cone_sim.CONE_R) for n, p in cone_xy.items()}
    world_all = walls + list(cone_objs.values())

    overrides = {'polar.steering_source': 'carrot', 'max_linear_vel': max_linear}
    if gt45:
        overrides.update({'goal_turn.enter_deg': 45.0, 'goal_turn.exit_deg': 15.0})
    params = cs.controller_params(overrides)
    model = replace(cs.RoverModel(), k_w_moving=k_w, heading_bias_deg=0.0)
    cone_sim.Drift.wheel_scale, cone_sim.Drift.gyro_scale, cone_sim.Drift.gyro_bias_deg_min = drift
    rover = cs.Rover(model, 0.0, 0.0, 0.0, rng=np.random.default_rng(seed + 1000))
    loc = cone_sim.Drift(model, rng)
    controller = mc.MotionController(params)
    shaper = mc.CommandShaper(0.3, 0.6, 0.6, 1.2)
    envelope = SafetyEnvelope()
    # the node's indoor settings; no attitude here (flat world): tilt off in the simulator
    safety = dict(SAFETY, **SAFETY_INDOOR)
    safety.update({'safety.tilt_enabled': False, 'safety.obstacle_enabled': stop,
                   'safety.stop_distance_m': stop_distance_m, 'safety.max_linear_vel': max_linear})
    km = KnownMap.from_yaml(TRACK)
    wl = WallLocalizer(km)
    goals = goals or [Goal(n, c, i + 1, is_start_cone=(n == 'Finish')) for i, (n, c, _) in enumerate(MISSION)]
    mp = dict(direction=direction)
    mp.update(mission_params or {})
    lpp = dict(LOCAL_INDOOR)
    lpp.update(lp_params or {})
    if not stuck:
        mp.setdefault('stuck_window_s', 1e9)
    mission = LoopPatrolMission(goals, km, (0.0, 0.0), 'red_orange', lpp, rec_params, **mp)

    straight = [(0.4 * (i + 1), 0.0, 1.0, 0.0) for i in range(8)]
    tick, out_period, cam_period, latency = 1.0 / params['tick_rate'], 0.1, 1.0 / 3.0, 0.17
    history, yaws = deque(), deque()
    results, hits, hit_now = [], [], False
    next_tick = next_out = next_cam = 0.0
    pending, model_cmd = None, (0.0, 0.0)
    confirm_at = None
    leg_t0 = 0.0
    min_clear = math.inf
    pose_err = []
    trace = []
    t, dt = 0.0, 0.02
    loc.update(t, dt, rover)
    while t < t_limit and not mission.done:
        loc.update(t, dt, rover)
        rx, ry, rth = loc.estimate()
        ex, ey, eth = wl.apply(rx, ry, rth) if localize else (rx, ry, rth)
        history.append((t, rover.x, rover.y, rover.theta))
        while len(history) > 1 and history[1][0] <= t - image_lag_s:
            history.popleft()
        yaws.append((t, eth))
        while len(yaws) > 1 and yaws[0][0] < t - 1.0:
            yaws.popleft()
        if t >= next_cam:
            next_cam += cam_period
            _, lx, ly, lth = history[0]
            pb, pf, pc = obs.profile(lx, ly, lth, world_all, rng)
            hit = pf < pc - 1e-3
            mission.observe_scan(t, ex, ey, eth, pb, pf, hit)
            if localize:
                wl.observe(ex, ey, eth, pb, pf, hit, mission.turn_rate_dps())
            envelope.observe_free_space(t, pb, np.where(hit, pf, 3.0), 3.0)
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
                                     r_true * (1 + rng.normal(0, 0.03)), t)
            pose_err.append(math.hypot(ex - rover.x, ey - rover.y))
        if t >= next_tick:
            next_tick += tick
            rate = 0.0
            if len(yaws) >= 2 and yaws[-1][0] - yaws[0][0] > 0.3:
                rate = abs(cs.mc.clip_angle(yaws[-1][1] - yaws[0][1])) / (yaws[-1][0] - yaws[0][0])
            if confirm_at is not None:
                if t >= confirm_at:
                    g = mission.goal
                    err = float(np.hypot(rover.x - cone_xy[g.name][0], rover.y - cone_xy[g.name][1])) if g.name in cone_xy else 99.0
                    if err <= JUDGE_M or t - leg_t0 > leg_limit_s:
                        results.append(dict(name=g.name, reached=err <= JUDGE_M, err=round(err, 2),
                                            time=round(t - leg_t0, 1)))
                        leg_t0 = t
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
                    err = float(np.hypot(rover.x - cone_xy[g.name][0], rover.y - cone_xy[g.name][1])) if g.name in cone_xy else 99.0
                    results.append(dict(name=g.name, reached=False, err=round(err, 2), time=round(t - leg_t0, 1)))
                    leg_t0 = t
                    mission.target = mission.target or None
                    mission.confirm(t, True)
                    controller.reset()
                elif step.command is not None:
                    controller.note_idle(t)
                    vv, ww, _h, _n = envelope.limit(t, step.command[0], step.command[1], safety)
                    pending = (t + latency, (vv, ww))
                elif step.carrot is not None:
                    cx, cy, _ = step.carrot
                    rel_x, rel_y = mc.robot_frame_offset(cx - ex, cy - ey, 90.0 - math.degrees(eth))
                    bearing = mc.goal_bearing_from_offset(rel_x, rel_y)
                    _m, vv, ww, _ = controller.command(t + latency, params, bearing, math.hypot(rel_x, rel_y),
                                                       straight, 4, True)
                    vv, ww, _h, _n = envelope.limit(t, vv, ww, safety)
                    pending = (t + latency, (vv, ww))
        if pending is not None and t >= pending[0]:
            model_cmd, pending = pending[1], None
        if t >= next_out:
            next_out += out_period
            lin, ang = shaper.step(model_cmd[0], model_cmd[1], out_period)
            rover.command(t, lin, ang)
        if record and (not trace or t - trace[-1][0] >= 0.5):
            trace.append((round(t, 1), mission.index, mission.state, round(rover.x, 2), round(rover.y, 2),
                          round(ex, 2), round(ey, 2), round(math.degrees(rover.theta)),
                          tuple(round(c, 2) for c in model_cmd)))
        rover.step(t, dt)
        c = obs.clearance((rover.x, rover.y), world_all)
        min_clear = min(min_clear, c)
        now_hit = c <= obs.FOOTPRINT_W / 2
        if now_hit and not hit_now:
            hits.append((round(t, 1), round(rover.x, 1), round(rover.y, 1)))
        hit_now = now_hit
        t += dt
    while len(results) < len(goals):
        results.append(dict(name=goals[len(results)].name, reached=False, err=float('nan'), time=float('nan')))
    return dict(layout=layout, world=world, seed=seed, reached=sum(r['reached'] for r in results),
                legs=len(goals), hits=len(hits), hit_at=hits, min_clear=min_clear, time=round(t, 1),
                results=results, replans=mission.replanner.replans, looks=mission.recovery.looks,
                pose_err_p50=float(np.median(pose_err)), pose_err_max=float(np.max(pose_err)),
                stucks=mission.stucks,
                notes=mission.notes, trace=trace, gaps=gaps,
                cones={k: tuple(np.round(v, 1)) for k, v in cone_xy.items()})


def _job(a):
    return run(**a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--layouts', default='loop,random')
    ap.add_argument('--worlds', default='tables,chairs,chairs+doors')
    ap.add_argument('--configs', default='stop+stuck,stop,nostop+stuck,nostop')
    ap.add_argument('--lag', type=float, default=0.8)
    ap.add_argument('--kw', type=float, default=1.1)
    ap.add_argument('--mission', default='{}')
    args = ap.parse_args()
    jobs = []
    for layout, w, cfg, seed in itertools.product(args.layouts.split(','), args.worlds.split(','),
                                                  args.configs.split(','), range(args.seeds)):
        jobs.append(dict(layout=layout, world=w, seed=seed, image_lag_s=args.lag, k_w=args.kw,
                         localize='nolocalize' not in cfg, stop=not cfg.startswith('nostop'),
                         stuck='stuck' in cfg, mission_params=json.loads(args.mission)))
    with Pool() as pool:
        res = pool.map(_job, jobs)
    for j, r in zip(jobs, res):
        r['cfg'] = ('stop' if j['stop'] else 'nostop') + ('+stuck' if j['stuck'] else '') \
            + ('' if j['localize'] else '+nolocalize')
    print(f"{'layout':7s} {'world':13s} {'cfg':12s} | {'cps ok':>7s} {'hits':>5s} {'w/hit':>6s} {'min_clr':>7s} "
          f"{'pose p50':>8s} {'pose max':>8s} {'time p50':>8s} {'replan':>6s} {'stuck':>5s}")
    key = lambda r: (r['layout'], r['world'], r['cfg'])  # noqa: E731
    for k, grp in itertools.groupby(sorted(res, key=key), key=key):
        g = list(grp)
        print(f"{k[0]:7s} {k[1]:13s} {k[2]:12s} | {sum(r['reached'] for r in g):3d}/{sum(r['legs'] for r in g):<3d} "
              f"{sum(r['hits'] for r in g):5d} {sum(1 for r in g if r['hits']):3d}/{len(g):<2d} "
              f"{min(r['min_clear'] for r in g):7.2f} {np.median([r['pose_err_p50'] for r in g]):8.2f} "
              f"{max(r['pose_err_max'] for r in g):8.2f} {np.median([r['time'] for r in g]):8.0f} "
              f"{np.mean([r['replans'] for r in g]):6.1f} {np.mean([r['stucks'] for r in g]):5.1f}")


if __name__ == '__main__':
    main()
