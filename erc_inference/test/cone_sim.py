#!/usr/bin/env python3
"""The NYU indoor loop driven cone to cone, in closed loop, with obstacles in the corridors.

Runs the robot's own code: cone_mission.ConeLeg (route, cone fusion, local
replanning, stop-and-look, search), local_planner.LocalReplanner,
sidestep.SideStep, motion_control.MotionController (steering_source carrot, as
carrot_controller_node forces) and CommandShaper, on controller_sim's rover
plant, with indoor_sim's dead reckoning and NYU walls.

Modelled, and where the numbers come from:
- Cones: the TRUE cone stands off the track file's position by N(0, 0.7 m)
  along the corridor and N(0, 0.25 m) across (the map is ~1 m accurate). The
  rover only has the track file.
- Cone detection (erc_perception.cones on rendered cones, full resolution):
  range <= 10 m, bearing error <= 1.4 deg, size-range error within +-6%.
  Emulated as p_detect 0.9 per frame inside +-55 deg, sigma 0.7 deg and 3%,
  occluded by walls and obstacles. Every cone of the target's colour class is
  visible (red and orange are one class).
- Free space: test/obstacles.profile (60 deg FOV, 3 m cap, degraded periphery).
- image_lag_s: profile and cone detections computed from where the rover WAS
  that long ago, integrated at where the estimate says it is now. 0.4-1.2 s in
  the field (camera content + SDK + tick + UniDepth).
Judged on the truth: a checkpoint counts if the rover stops within judge_m of
the true cone; a hit is the rover's centre within FOOTPRINT_W/2 of a wall, a
table, an obstacle or a cone.

    source /opt/ros/jazzy/setup.bash && source ~/lsdc_ws/install/setup.bash
    python3 test/cone_sim.py [--seeds 4]
"""
import argparse
import itertools
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
import indoor_sim as isim  # noqa: E402
import obstacles as obs  # noqa: E402
from erc_inference import indoor_track as it  # noqa: E402
from erc_inference import motion_control as mc  # noqa: E402
from erc_inference.cone_mission import ConeLeg  # noqa: E402
from erc_inference.safety_envelope import DEFAULTS as SAFETY, SafetyEnvelope  # noqa: E402
from erc_inference.sidestep import SideStep  # noqa: E402

CONE_CLASS = {'red': 'red_orange', 'orange': 'red_orange', 'blue': 'blue', 'green': 'green', 'yellow': 'yellow'}
CONE_R = 0.09
JUDGE_M = 2.0


def box(cx, cy, w, h):
    S = obs.Segment
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    return [S((x0, y0), (x1, y0)), S((x1, y0), (x1, y1)), S((x1, y1), (x0, y1)), S((x0, y1), (x0, y0))]


SCENARIOS = {
    # walls + the five tables of the south corridor, nothing else
    'tables': [],
    # one thing in each corridor, placed on or next to the route
    'clutter': (box(13.0, 0.3, 0.6, 0.6)                      # a box in the south corridor
                + [obs.Disc((36.3, 5.5), 0.25)]               # a person in the 1.9 m east corridor
                + [obs.Disc((28.0, 27.9), 0.30)]              # a chair in the north corridor
                + box(9.55, 21.0, 0.5, 0.8)),                 # a cart against the west wall
}

# cones: cone detection refines the goal; lp: LocalReplanner; ss: SideStep reflex;
# stop: safety_envelope obstacle stop (linear 0, keep turning) instead of SideStep;
# look: stop-and-look before a detour; gt45: turn in place when the carrot is >45 deg off
_V = dict(cones=False, lp=False, ss=False, look=False, stop=False, gt45=False)
VARIANTS = {name: dict(_V, **{k: True for k in name.split('+') if k in _V})
            for name in ['prior', 'cones', 'cones+ss', 'cones+lp', 'cones+ss+lp', 'cones+ss+lp+look',
                         'cones+gt45', 'cones+lp+stop', 'cones+lp+stop+gt45', 'cones+lp+stop+look+gt45']}


class Drift(isim.DeadReckoning):
    pass


def true_cones(track, rng):
    """Where the cones really stand: the track file's point, moved along its corridor."""
    out = {}
    for c in track.checkpoints:
        # corridor direction at the cone: the leg's last segment
        prev = c.via[-1] if c.via else None
        if prev is None:
            d = np.array([1.0, 0.0])
        else:
            d = np.array([c.x - prev[0], c.y - prev[1]], float)
            d = d / max(np.hypot(*d), 1e-6) if np.hypot(*d) > 1e-6 else np.array([1.0, 0.0])
        n = np.array([-d[1], d[0]])
        along, across = rng.normal(0, 0.7), float(np.clip(rng.normal(0, 0.25), -0.5, 0.5))
        out[c.sequence] = np.array([c.x, c.y]) + along * d + across * n
    return out


def visible(o, target, world, max_t):
    d = np.asarray(target, float) - o
    dist = float(np.hypot(*d))
    if dist < 1e-6 or dist > max_t:
        return dist <= max_t
    u = d / dist
    return all(w.ray_hit(o, u, dist - CONE_R) == math.inf for w in world)


def run(scenario, variant, seed, drift=(0.01, 0.01, 0.5), k_w=1.1, image_lag_s=0.0,
        lp_params=None, leg_params=None, record=False, t_limit=900.0):
    v = VARIANTS[variant]
    rng = np.random.default_rng(seed)
    track = it.load_track(isim.TRACK_FILE)
    cones = true_cones(track, rng)
    cone_objs = {s: obs.Disc(p, CONE_R) for s, p in cones.items()}
    walls = isim.nyu_walls() + SCENARIOS[scenario]
    world_all = walls + list(cone_objs.values())
    overrides = {'polar.steering_source': 'carrot'}
    if v['gt45']:
        overrides.update({'goal_turn.enter_deg': 45.0, 'goal_turn.exit_deg': 15.0})
    params = cs.controller_params(overrides)
    envelope = SafetyEnvelope() if v['stop'] else None
    safety = dict(SAFETY, **{'safety.tilt_enabled': False, 'safety.obstacle_enabled': True})
    stops = 0
    model = replace(cs.RoverModel(), k_w_moving=k_w, heading_bias_deg=0.0)
    Drift.wheel_scale, Drift.gyro_scale, Drift.gyro_bias_deg_min = drift
    s0 = track.start
    rover = cs.Rover(model, s0.x, s0.y, s0.yaw, rng=np.random.default_rng(seed + 1000))
    loc = Drift(model, rng)
    controller = mc.MotionController(params)
    shaper = mc.CommandShaper(0.3, 0.6, 0.6, 1.2)          # indoor_mission_node defaults
    sidestep = SideStep() if v['ss'] else None
    lp_kw = dict(lp_params or {})

    straight = [(0.4 * (i + 1), 0.0, 1.0, 0.0) for i in range(8)]
    tick, out_period, cam_period, latency = 1.0 / params['tick_rate'], 0.1, 1.0 / 3.0, 0.17
    history = deque()          # (t, x, y, theta) truth, for image lag
    leg_i, leg, leg_t0 = 0, None, 0.0
    confirm_until = None
    results, hits, hit_now = [], [], False
    next_tick = next_out = next_cam = 0.0
    pending, model_cmd, override = None, (0.0, 0.0), None
    replans = looks = sidesteps = 0
    reached_est = []
    min_clear = math.inf
    t, dt = 0.0, 0.02
    trace = []
    carrot = (s0.x, s0.y, 0.0)
    loc.update(t, dt, rover)
    while t < t_limit and leg_i < len(track.checkpoints):
        loc.update(t, dt, rover)
        ex, ey, eth = loc.estimate()
        history.append((t, rover.x, rover.y, rover.theta))
        while len(history) > 1 and history[1][0] <= t - image_lag_s:
            history.popleft()
        cp = track.checkpoints[leg_i]
        if leg is None and confirm_until is None:
            leg = ConeLeg(track.leg_route(cp, ex, ey), (cp.x, cp.y), cp.cone, lp_kw,
                          known_cones=reached_est[-1:], look_enabled=v['look'],
                          search_enabled=v['cones'], **(leg_params or {}))
            leg_t0 = t
            if sidestep:
                sidestep.reset()
        if confirm_until is not None:
            if t >= confirm_until:
                confirm_until = None
                leg_i += 1
            override = (0.0, 0.0)
        # ---- camera / perception, 3 Hz, from the lagged true pose ----
        if leg is not None and t >= next_cam:
            next_cam += cam_period
            _, lx, ly, lth = history[0]
            if v['lp']:
                pb, pf, pc = obs.profile(lx, ly, lth, world_all, rng)
                leg.observe_scan(t, ex, ey, eth, pb, pf, pf < pc - 1e-3)
            if v['cones']:
                o = np.array([lx, ly])
                for s, cpos in cones.items():
                    if CONE_CLASS[track.checkpoint(s).cone] != CONE_CLASS[cp.cone]:
                        continue
                    d = cpos - o
                    rng_true = float(np.hypot(*d))
                    brg = math.degrees(cs.mc.clip_angle(math.atan2(d[1], d[0]) - lth))
                    if abs(brg) > 55.0 or rng_true > 10.0 or rng.random() > 0.9:
                        continue
                    if not visible(o, cpos, walls + [c for k, c in cone_objs.items() if k != s], 10.0):
                        continue
                    leg.observe_cone(ex, ey, eth, brg + rng.normal(0, 0.7), rng_true * (1 + rng.normal(0, 0.03)))
        # ---- the mission node: carrot / override, at the tick ----
        if leg is not None and t >= next_tick:
            step = leg.update(t, ex, ey, eth)
            override = step.command
            if step.arrived or t - leg_t0 > 300.0:
                true_cone = cones[cp.sequence]
                results.append(dict(seq=cp.sequence, reached=bool(step.arrived),
                                    err=float(np.hypot(rover.x - true_cone[0], rover.y - true_cone[1])),
                                    seen=leg.target.confirmed, time=t - leg_t0, note=step.note,
                                    log=list(leg.notes)))
                replans += leg.replanner.replans
                looks += leg.looks
                if leg.target.confirmed:
                    reached_est.append(leg.target.estimate())
                leg = None
                confirm_until = t + 1.5
                override = (0.0, 0.0)
                controller.reset()
            carrot = step.carrot
        # ---- the model-free controller, 3 Hz ----
        if t >= next_tick:
            next_tick += tick
            if leg is None or override is not None:
                model_cmd = (0.0, 0.0) if override is None else override
                controller.note_idle(t)
                pending = None
                if override is not None and override != (0.0, 0.0):
                    pending = (t + latency, override)
            else:
                heading_deg = 90.0 - math.degrees(eth)
                rel_x, rel_y = mc.robot_frame_offset(carrot[0] - ex, carrot[1] - ey, heading_deg)
                bearing = mc.goal_bearing_from_offset(rel_x, rel_y)
                _mode, vv, ww, _ = controller.command(t + latency, params, bearing, math.hypot(rel_x, rel_y),
                                                      straight, 4, True)
                if sidestep is not None:
                    _, lx, ly, lth = history[0]
                    ob, of, _oc = obs.profile(lx, ly, lth, world_all, rng)
                    state0 = sidestep.state
                    vv, ww, _note, resumed = sidestep.step(t, eth, (ex, ey), ob, of, vv, ww, bearing)
                    if state0 == 'follow' and sidestep.state != 'follow':
                        sidesteps += 1
                    if resumed:
                        controller.reset()
                if envelope is not None:
                    _, lx, ly, lth = history[0]
                    eb, ef, ec = obs.profile(lx, ly, lth, world_all, rng)
                    envelope.observe_free_space(t, eb, np.where(ef < ec - 1e-3, ef, 3.0), 3.0)
                    was = envelope._obstacle_stop
                    vv, ww, _hard, _n = envelope.limit(t, vv, ww, safety)
                    stops += int(envelope._obstacle_stop and not was)
                pending = (t + latency, (vv, ww))
        if pending is not None and t >= pending[0]:
            model_cmd, pending = pending[1], None
        # ---- indoor_mission_node output, 10 Hz ----
        if t >= next_out:
            next_out += out_period
            target = model_cmd if override is None else override
            lin, ang = shaper.step(target[0], target[1], out_period)
            if override == (0.0, 0.0):
                shaper.reset()
                lin, ang = 0.0, 0.0
            rover.command(t, lin, ang)
        if record and (not trace or t - trace[-1][0] >= 0.5):
            trace.append((round(t, 1), leg_i, round(rover.x, 2), round(rover.y, 2), round(ex, 2), round(ey, 2),
                          round(math.degrees(rover.theta)), override, sidestep.state if sidestep else '',
                          tuple(round(q, 1) for q in carrot[:2]) if leg is not None else None,
                          round(model_cmd[0], 2), round(model_cmd[1], 2)))
        rover.step(t, dt)
        c = obs.clearance((rover.x, rover.y), world_all)
        min_clear = min(min_clear, c)
        now_hit = c <= obs.FOOTPRINT_W / 2
        if now_hit and not hit_now:
            hits.append((round(t, 1), round(rover.x, 1), round(rover.y, 1)))
        hit_now = now_hit
        t += dt
    reached = sum(1 for r in results if r['reached'] and r['err'] <= JUDGE_M)
    return dict(scenario=scenario, variant=variant, seed=seed, k_w=k_w, lag=image_lag_s,
                reached=reached, legs=len(track.checkpoints), hits=len(hits), hit_at=hits,
                min_clear=min_clear, time=t, replans=replans, looks=looks, sidesteps=sidesteps + stops,
                errs=[round(r['err'], 2) for r in results], seen=[r['seen'] for r in results],
                notes=[r['note'] for r in results], logs=[r['log'] for r in results], trace=trace)


def _job(a):
    return run(**a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--scenarios', default='tables,clutter')
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--lags', default='0.0,0.8')
    ap.add_argument('--kw', default='1.1')
    ap.add_argument('--lp', default='{}', help='LocalReplanner overrides, JSON')
    args = ap.parse_args()
    import json
    lp_params = json.loads(args.lp)
    jobs = [dict(scenario=s, variant=v, seed=seed, image_lag_s=float(lag), k_w=float(kw), lp_params=lp_params)
            for s, v, lag, kw, seed in itertools.product(args.scenarios.split(','), args.variants.split(','),
                                                         args.lags.split(','), args.kw.split(','),
                                                         range(args.seeds))]
    with Pool() as pool:
        res = pool.map(_job, jobs)
    print(f"{'scenario':8s} {'variant':17s} {'lag':>4s} {'k_w':>4s} | {'cones ok':>8s} {'hits':>5s} "
          f"{'runs w/ hit':>11s} {'min_clr':>7s} {'err p50':>7s} {'err max':>7s} {'time':>6s} {'replan':>6s} "
          f"{'looks':>5s} {'ss':>4s}")
    key = lambda r: (r['scenario'], r['variant'], r['lag'], r['k_w'])  # noqa: E731
    for k, grp in itertools.groupby(sorted(res, key=key), key=key):
        g = list(grp)
        errs = [e for r in g for e in r['errs']]
        print(f"{k[0]:8s} {k[1]:17s} {k[2]:4.1f} {k[3]:4.2f} | {sum(r['reached'] for r in g):3d}/{sum(r['legs'] for r in g):<4d} "
              f"{sum(r['hits'] for r in g):5d} {sum(1 for r in g if r['hits']):5d}/{len(g):<5d} "
              f"{min(r['min_clear'] for r in g):7.2f} {np.median(errs):7.2f} {max(errs):7.2f} "
              f"{np.median([r['time'] for r in g]):6.0f} {np.mean([r['replans'] for r in g]):6.1f} "
              f"{np.mean([r['looks'] for r in g]):5.1f} {np.mean([r['sidesteps'] for r in g]):4.1f}")


if __name__ == '__main__':
    main()
