#!/usr/bin/env python3
"""Off-road image-goal mission in a synthetic arena: find where the goal photo was taken.

Runs homing_mission.HomingMission (LocalReplanner and BlockedRecovery inside) with
goal_homing.GoalMatcher.solve -- the real geometry -- MotionController (carrot, as
carrot_controller_node), SafetyEnvelope and CommandShaper, on controller_sim's plant with
indoor_sim's dead reckoning, as loop_patrol_sim.py.

The arena (Verti-Arena-like, arxiv 2508.08226): 8 x 8 m, walls round it with textured lab
equipment behind, boulders (discs, r 0.2-0.5 m, as tall as they are wide) that block the
way and the view. What the camera sees is modelled, not rendered: scene points (on the
walls, on the boulders, gravel on the ground), projected through the TAREA1 fisheye model at
13.2 cm; a point counts in a view if it is in the image, within range and not behind a
boulder. A match is a point seen in both the goal view and the current one, kept with a
probability that falls with the change of viewpoint, plus random false matches, with pixel
noise. Relief, slopes and traction are NOT modelled (flat floor, no tilt).

The judge: the rover's true position within judge_m of where the photo was taken when the
mission says ARRIVED (the SDK's real criterion is unknown).

    python3 test/homing_sim.py [--seeds 20]
"""
import argparse
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '..', '..', 'erc_perception'))

import controller_sim as cs  # noqa: E402
import cone_sim  # noqa: E402
import obstacles as obs  # noqa: E402
from erc_inference import motion_control as mc  # noqa: E402
from erc_inference.goal_homing import Camera, GoalMatcher  # noqa: E402
from erc_inference.homing_mission import HomingGoal, HomingMission  # noqa: E402
from erc_inference.safety_envelope import DEFAULTS as SAFETY, SafetyEnvelope  # noqa: E402

try:
    from erc_perception.geometry import _distort
except ImportError:
    def _distort(x_u, y_u, lam):
        r_u = np.hypot(x_u, y_u)
        disc = 1.0 - 4.0 * lam * r_u ** 2
        with np.errstate(invalid='ignore', divide='ignore'):
            r_d = np.where(r_u > 1e-12, (1.0 - np.sqrt(np.clip(disc, 0.0, None))) / (2.0 * lam * r_u), 0.0)
            s = np.where(r_u > 1e-12, r_d / np.maximum(r_u, 1e-12), 1.0)
        return x_u * np.where(disc < 0, np.nan, s), y_u * np.where(disc < 0, np.nan, s)

CAM = Camera()
CAM_H = 0.132
JUDGE_M = 0.5
ARENA = (-1.0, 7.0, -4.0, 4.0)         # x0, x1, y0, y1; the rover starts at (0, 0) facing +x
SAFETY_OFFROAD = {'safety.max_linear_vel': 0.25, 'safety.max_reverse_vel': 0.25, 'safety.obstacle_enabled': False,
                  'safety.tilt_enabled': False, 'safety.stop_if_attitude_stale': False}
LOCAL_OFFROAD = {'inflate_m': 0.3, 'unseen_extend_m': 0.1}


class Scene:
    def __init__(self, rng, n_boulders=10, n_wall=900, n_gravel=500):
        x0, x1, y0, y1 = ARENA
        self.boulders = []
        while len(self.boulders) < n_boulders:
            c = np.array([rng.uniform(x0 + 0.8, x1 - 0.5), rng.uniform(y0 + 0.5, y1 - 0.5)])
            r = rng.uniform(0.2, 0.5)
            if np.hypot(*c) < 1.2 or any(np.hypot(*(c - b.c)) < b.r + r + 0.5 for b in self.boulders):
                continue
            self.boulders.append(obs.Disc(c, r))
        self.walls = [obs.Segment((x0, y0), (x1, y0)), obs.Segment((x1, y0), (x1, y1)),
                      obs.Segment((x1, y1), (x0, y1)), obs.Segment((x0, y1), (x0, y0))]
        pts, owner = [], []
        # lab equipment and walls: 1.5 m behind the arena's edge, 0-2.5 m high
        for k in range(n_wall):
            side = rng.integers(4)
            u = rng.uniform(0, 1)
            if side == 0:
                p = (x0 - 1.5 + u * (x1 - x0 + 3), y0 - 1.5)
            elif side == 1:
                p = (x1 + 1.5, y0 - 1.5 + u * (y1 - y0 + 3))
            elif side == 2:
                p = (x0 - 1.5 + u * (x1 - x0 + 3), y1 + 1.5)
            else:
                p = (x0 - 1.5, y0 - 1.5 + u * (y1 - y0 + 3))
            pts.append((p[0], p[1], rng.uniform(0.0, 2.5)))
            owner.append(-1)
        for i, b in enumerate(self.boulders):
            for _ in range(int(60 * b.r / 0.35)):
                a = rng.uniform(0, 2 * math.pi)
                el = rng.uniform(0, 0.5 * math.pi)
                pts.append((b.c[0] + b.r * math.cos(a) * math.cos(el), b.c[1] + b.r * math.sin(a) * math.cos(el),
                            b.r * math.sin(el)))
                owner.append(i)
        for _ in range(n_gravel):
            pts.append((rng.uniform(x0, x1), rng.uniform(y0, y1), 0.0))
            owner.append(-2)
        self.P = np.array(pts, float)
        self.owner = np.array(owner)
        self.world = self.walls + self.boulders

    def free_pose(self, rng):
        x0, x1, y0, y1 = ARENA
        while True:
            p = np.array([rng.uniform(x0 + 0.6, x1 - 0.6), rng.uniform(y0 + 0.6, y1 - 0.6)])
            if obs.clearance(p, self.boulders) > 0.6:
                return p

    def view(self, x, y, th, max_range=12.0):
        """Pixels (N, 2) of the visible scene points and their indices."""
        d = self.P[:, :2] - (x, y)
        c, s = math.cos(th), math.sin(th)
        fwd = d[:, 0] * c + d[:, 1] * s
        left = -d[:, 0] * s + d[:, 1] * c
        up = self.P[:, 2] - CAM_H
        ok = (fwd > 0.2) & (np.hypot(fwd, left) < max_range)
        xn, yn = np.full(len(d), np.nan), np.full(len(d), np.nan)
        xn[ok], yn[ok] = -left[ok] / fwd[ok], -up[ok] / fwd[ok]
        xd, yd = _distort(xn, yn, CAM.lam)
        u, v = CAM.cx + CAM.f * xd, CAM.cy + CAM.f * yd
        ok &= np.isfinite(u) & (u >= 0) & (u < CAM.width) & (v >= 0) & (v < CAM.height_px)
        # hidden behind a boulder: the sight line passes over it lower than its top
        dist = np.hypot(d[:, 0], d[:, 1])
        for i, b in enumerate(self.boulders):
            bc = b.c - (x, y)
            t = (d[:, 0] * bc[0] + d[:, 1] * bc[1]) / np.maximum(dist, 1e-9)       # along the ray
            perp = np.abs(d[:, 0] * bc[1] - d[:, 1] * bc[0]) / np.maximum(dist, 1e-9)
            h_ray = CAM_H + (self.P[:, 2] - CAM_H) * np.clip(t / np.maximum(dist, 1e-9), 0, 1)
            hide = ok & (self.owner != i) & (perp < b.r) & (t > 0) & (t < dist - 0.05) & (h_ray < b.r)
            ok &= ~hide
        idx = np.flatnonzero(ok)
        return np.stack([u[idx], v[idx]], 1), idx


MATCH_QUALITY = 1.0     # scales SIFT repeatability (a sweep knob)


def synth_matches(scene, goal_view, cur_pose, goal_pose, rng, noise_px=1.0, outlier_frac=0.15, max_n=400):
    gu, gidx = goal_view
    cu, cidx = scene.view(*cur_pose)
    common, gi, ci = np.intersect1d(gidx, cidx, return_indices=True)
    dyaw = abs(math.degrees(cs.mc.clip_angle(cur_pose[2] - goal_pose[2])))
    dd = math.hypot(cur_pose[0] - goal_pose[0], cur_pose[1] - goal_pose[1])
    # SIFT repeatability falls with the viewpoint change; gravel matches worst (self-similar)
    keep = MATCH_QUALITY * 0.6 * math.exp(-dyaw / 50.0) * math.exp(-dd / 6.0)
    pk = np.where(scene.owner[common] == -2, 0.3 * keep, keep)
    sel = rng.random(len(common)) < pk
    a, b = cu[ci[sel]], gu[gi[sel]]
    if len(a) > max_n:
        k = rng.choice(len(a), max_n, replace=False)
        a, b = a[k], b[k]
    n_out = int(outlier_frac * len(a)) + int(rng.integers(0, 6))
    if n_out and len(cu) and len(gu):
        a = np.vstack([a, cu[rng.integers(0, len(cu), n_out)]])
        b = np.vstack([b, gu[rng.integers(0, len(gu), n_out)]])
    return a + rng.normal(0, noise_px, a.shape), b + rng.normal(0, noise_px, b.shape)


def run(seed=0, n_boulders=10, drift=(0.02, 0.02, 1.0), k_w=1.1, image_lag_s=0.8, t_limit=900.0,
        goal_ahead=None, mission_params=None, record=False, match_quality=1.0):
    global MATCH_QUALITY
    MATCH_QUALITY = match_quality
    rng = np.random.default_rng(seed)
    scene = Scene(rng, n_boulders)
    g = scene.free_pose(rng)
    if goal_ahead is not None:            # difficulty: the goal in view from the start or not
        while (abs(math.atan2(g[1], g[0])) < math.radians(50)) != goal_ahead:
            g = scene.free_pose(rng)
    gth = rng.uniform(-math.pi, math.pi)
    goal_pose = (float(g[0]), float(g[1]), gth)
    goal_view = scene.view(*goal_pose)
    matcher = GoalMatcher.__new__(GoalMatcher)     # geometry only: no image, no SIFT
    import cv2
    from erc_inference.goal_homing import DEFAULTS as HD
    matcher.cv2, matcher.cam, matcher.p, matcher.warning = cv2, CAM, dict(HD), ''

    params = cs.controller_params({'polar.steering_source': 'carrot', 'max_linear_vel': 0.25,
                                   'goal_turn.enter_deg': 45.0, 'goal_turn.exit_deg': 15.0})
    model = replace(cs.RoverModel(), k_w_moving=k_w, heading_bias_deg=0.0)
    cone_sim.Drift.wheel_scale, cone_sim.Drift.gyro_scale, cone_sim.Drift.gyro_bias_deg_min = drift
    rover = cs.Rover(model, 0.0, 0.0, 0.0, rng=np.random.default_rng(seed + 1000))
    loc = cone_sim.Drift(model, rng)
    controller = mc.MotionController(params)
    shaper = mc.CommandShaper(0.3, 0.6, 0.6, 1.2)
    envelope = SafetyEnvelope()
    safety = dict(SAFETY, **SAFETY_OFFROAD)
    mission = HomingMission([HomingGoal('G1', 'sim', 1)], (0.0, 0.0), dict(LOCAL_OFFROAD), None,
                            **(mission_params or {}))

    straight = [(0.4 * (i + 1), 0.0, 1.0, 0.0) for i in range(8)]
    tick, out_period, cam_period, latency = 1.0 / params['tick_rate'], 0.1, 1.0 / 3.0, 0.17
    history = deque()
    yaws = deque()
    hits, hit_now = [], False
    next_tick = next_out = next_cam = 0.0
    pending, model_cmd = None, (0.0, 0.0)
    min_clear = math.inf
    trace = []
    arrived_at, err = None, float('nan')
    stop_pose = None
    t, dt = 0.0, 0.02
    loc.update(t, dt, rover)
    while t < t_limit:
        loc.update(t, dt, rover)
        ex, ey, eth = loc.estimate()
        history.append((t, rover.x, rover.y, rover.theta, ex, ey, eth))
        while len(history) > 1 and history[1][0] <= t - image_lag_s:
            history.popleft()
        yaws.append((t, eth))
        while len(yaws) > 1 and yaws[0][0] < t - 1.0:
            yaws.popleft()
        if t >= next_cam:
            next_cam += cam_period
            tl, lx, ly, lth, lex, ley, leth = history[0]        # what the picture shows, and its pose
            pb, pf, pc = obs.profile(lx, ly, lth, scene.world, rng)
            hit = pf < pc - 1e-3
            mission.observe_scan(t, lex, ley, leth, pb, pf, hit)
            envelope.observe_free_space(t, pb, np.where(hit, pf, 3.0), 3.0)
            a, b = synth_matches(scene, goal_view, (lx, ly, lth), goal_pose, rng)
            if len(a) >= 5:
                h = matcher.solve(matcher._normalize(a, CAM.width), matcher._normalize(b, CAM.width))
            else:
                from erc_inference.goal_homing import Homing
                h = Homing(matches=len(a), note='few')
            mission.observe_homing(tl, lex, ley, leth, h)
            standing = mission.state == 'measure' and mission._stop_t0 is not None \
                and tl >= mission._stop_t0 + mission.p['settle_s']
            if standing and mission.wants_motion_check and stop_pose is not None:
                a2, b2 = synth_matches(scene, scene.view(*stop_pose), (lx, ly, lth), stop_pose, rng)
                h2 = matcher.solve(matcher._normalize(a2, CAM.width), matcher._normalize(b2, CAM.width)) \
                    if len(a2) >= 5 else None
                mission.observe_motion(tl, h2.parallax_px if h2 is not None and h2.ok else float('nan'))
            elif standing and not mission.wants_motion_check:
                stop_pose = (lx, ly, lth)
        if t >= next_tick:
            next_tick += tick
            rate = 0.0
            if len(yaws) >= 2 and yaws[-1][0] - yaws[0][0] > 0.3:
                rate = abs(cs.mc.clip_angle(yaws[-1][1] - yaws[0][1])) / (yaws[-1][0] - yaws[0][0])
            step = mission.update(t, ex, ey, eth, rate)
            if step.arrived:
                arrived_at = t
                err = math.hypot(rover.x - goal_pose[0], rover.y - goal_pose[1])
                break
            if step.command is not None:
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
            trace.append((round(t, 1), mission.state, round(rover.x, 2), round(rover.y, 2),
                          round(math.degrees(rover.theta)), tuple(round(c, 2) for c in model_cmd)))
        px, py = rover.x, rover.y
        rover.step(t, dt)
        if obs.clearance((rover.x, rover.y), scene.world) <= obs.FOOTPRINT_W / 2 - 0.02:
            rover.x, rover.y = px, py      # against a wall or a boulder: the wheels spin, it does not pass
        c = obs.clearance((rover.x, rover.y), scene.world)
        min_clear = min(min_clear, c)
        now_hit = c <= obs.FOOTPRINT_W / 2
        if now_hit and not hit_now:
            hits.append((round(t, 1), round(rover.x, 1), round(rover.y, 1)))
        hit_now = now_hit
        t += dt
    final = math.hypot(rover.x - goal_pose[0], rover.y - goal_pose[1])
    return dict(seed=seed, goal=tuple(np.round(goal_pose[:2], 1)), goal_yaw=round(math.degrees(gth)),
                start_dist=round(math.hypot(*goal_pose[:2]), 1), arrived=arrived_at is not None,
                ok=arrived_at is not None and err <= JUDGE_M, err=round(err if err == err else final, 2),
                time=round(arrived_at if arrived_at is not None else t, 1), hits=len(hits), stops=mission.stops,
                stucks=mission.stucks,
                min_clear=round(min_clear, 2), notes=mission.notes, trace=trace)


def _job(a):
    return run(**a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--boulders', type=int, default=10)
    ap.add_argument('--lag', type=float, default=0.8)
    ap.add_argument('--drift', default='0.02,0.02,1.0')
    args = ap.parse_args()
    drift = tuple(float(v) for v in args.drift.split(','))
    jobs = [dict(seed=s, n_boulders=args.boulders, image_lag_s=args.lag, drift=drift) for s in range(args.seeds)]
    with Pool() as pool:
        res = pool.map(_job, jobs)
    for r in res:
        print(f"seed {r['seed']:3d} goal {r['goal']} ({r['start_dist']:4.1f} m, yaw {r['goal_yaw']:+4d}) "
              f"{'OK ' if r['ok'] else ('ARR' if r['arrived'] else '---')} err {r['err']:5.2f} m "
              f"t {r['time']:6.1f} s stops {r['stops']:3d} stuck {r['stucks']} hits {r['hits']}")
    ok = [r for r in res if r['ok']]
    print(f"within {JUDGE_M} m: {len(ok)}/{len(res)}; arrived (any error): {sum(r['arrived'] for r in res)}; "
          f"median time of successes {np.median([r['time'] for r in ok]) if ok else float('nan'):.0f} s; "
          f"median error when arrived {np.median([r['err'] for r in res if r['arrived']]) if any(r['arrived'] for r in res) else float('nan'):.2f} m")


if __name__ == '__main__':
    main()
