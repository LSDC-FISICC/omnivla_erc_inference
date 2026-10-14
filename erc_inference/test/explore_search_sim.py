#!/usr/bin/env python3
"""Search mission with no known map (erc_inference.explore_search), in closed loop.

A small floor the mission knows nothing about: an L of two 2 m corridors and a 5 x 4 m room
off the first one, potted plants (discs: they are obstacles too) standing in it. The rover
gets what it would get on the real one: obstacles.profile() as /erc/free_space (its measured
periphery, misses and noise), plant detections with bearing and range noise and a line of
sight (range a fraction short and 8% noisy, as the depth model's; now and then a detection that
is nothing), a pose that drifts (wheel and gyro scale errors), and its commands arrive late. A
plain carrot follower drives (v 0.25 m/s, turn in place beyond 45 deg): not MotionController
and not controller_sim's plant, so this checks the mission's logic and map, not the tuning.

    python3 test/explore_search_sim.py [--seeds 6] [--plot out.png]
"""
import argparse
import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import obstacles as obs  # noqa: E402
from erc_inference.explore_search import ExploreSearchMission  # noqa: E402
from erc_inference.image_goal_mission import Goal  # noqa: E402
from erc_inference.loop_patrol import LOCAL_INDOOR, PATROL_COMPETITION  # noqa: E402

PLANT_R = 0.2


def world():
    """Walls of the floor (m). The rover starts at (0, 0) facing +x, in the first corridor."""
    W = lambda a, b: obs.Segment(a, b)
    return [
        # corridor A: x -1.5..14, y -1..1; door to the room at x 5..6.2 on its north wall
        W((-1.5, -1.0), (14.0, -1.0)), W((-1.5, -1.0), (-1.5, 1.0)),
        W((-1.5, 1.0), (5.0, 1.0)), W((6.2, 1.0), (12.0, 1.0)),
        # corridor B: x 12..14, y 1..11, closed at the top
        W((14.0, -1.0), (14.0, 11.0)), W((12.0, 1.0), (12.0, 11.0)), W((12.0, 11.0), (14.0, 11.0)),
        # the room: x 3..8, y 1..5 (its south wall is corridor A's north wall)
        W((3.0, 1.0), (3.0, 5.0)), W((3.0, 5.0), (8.0, 5.0)), W((8.0, 5.0), (8.0, 1.0)),
    ]


PLANTS = np.array([[9.0, 0.6], [6.8, 4.2], [13.4, 7.0], [3.6, 4.4]])
if os.environ.get('PLANTS_XY'):          # "x,y;x,y;..." -- another arrangement, e.g. one in view from the start
    PLANTS = np.array([[float(v) for v in q.split(',')] for q in os.environ['PLANTS_XY'].split(';')])


def run(seed, n_goals=3, t_max=1800.0, delay_s=1.0, wheel_scale=0.97, gyro_scale=1.03, record=False, count=False,
        reactive=False):
    rng = np.random.default_rng(seed)
    walls = world()
    things = walls + [obs.Disc(tuple(p), PLANT_R) for p in PLANTS]
    kw = dict(PATROL_COMPETITION)
    kw['max_shows'] = 0
    if count:                        # driving past and counting, as plants_controller.launch.py runs it
        kw.update(start_scan=False, count_on_sight=True, forward_first=True, look_around=False,
                  same_class_apart_m=1.5)
    if reactive:
        kw['drive'] = 'reactive'
    m = ExploreSearchMission([Goal(f'Maceta{i + 1}', 'potted plant', i + 1) for i in range(n_goals)],
                             (0.0, 0.0), dict(LOCAL_INDOOR), **kw)
    x = y = th = 0.0                 # truth
    ex = ey = eth = 0.0              # what the odometry says
    dt, t = 0.1, 0.0
    queue = deque([(0.0, 0.0)] * int(round(delay_s / dt)))
    v = w = 0.0
    hits, shown, trail, step = 0, [], [], None
    in_contact = False
    while t < t_max and not m.done:
        k = int(round(t / dt))
        if k % 3 == 0:               # perception and the mission at ~3 Hz
            b, free, caps = obs.profile(x, y, th, things, rng)
            m.observe_scan(t, ex, ey, eth, b, np.minimum(free, obs.MAX_RANGE_M), free < caps - 0.05)
            frame = []
            for px, py in PLANTS:
                r = math.hypot(px - x, py - y)
                brg = math.degrees((math.atan2(py - y, px - x) - th + math.pi) % (2 * math.pi) - math.pi)
                d = np.array([px - x, py - y]) / max(r, 1e-6)
                blocked = min((s.ray_hit(np.array([x, y]), d, r) for s in walls), default=math.inf) < r - 0.05
                if 0.4 < r < 8.0 and abs(brg) < 50 and not blocked and rng.random() < 0.8:
                    # the depth model's range: a fraction short (4.4 m for ~4.9 m on the rover) and noisy
                    frame.append(('potted plant', brg + rng.normal(0, 1.5), r * (0.92 + rng.normal(0, 0.08))))
            if rng.random() < 0.02:          # a detection that is not a plant, once in a while
                frame.append(('potted plant', rng.uniform(-45, 45), rng.uniform(1.5, 7.0)))
            m.observe_objects(ex, ey, eth, frame, t)
            step = m.update(t, ex, ey, eth, abs(w))
            for c in m.cones.clusters:       # counted while driving past: which true plant, how far the rover is
                if count and c.name.startswith('cone') and c.id not in {s[3] for s in shown}:
                    which = int(np.hypot(*(PLANTS - c.estimate()).T).argmin())
                    shown.append((round(t), round(float(np.hypot(*(PLANTS[which] - (x, y)))), 2), which, c.id,
                                  round(float(np.hypot(*(PLANTS[which] - c.estimate()))), 2)))
            if step.arrived:
                e = m.target.estimate()
                which = int(np.hypot(*(PLANTS - e).T).argmin())          # the true plant this marker is
                shown.append((round(t), round(float(np.hypot(*(PLANTS[which] - (x, y)))), 2), which, m.target.id,
                              round(float(np.hypot(*(PLANTS[which] - e))), 2)))
                m.confirm(t, True)
                cmd = (0.0, 0.0)
            elif step.command is not None:
                cmd = step.command
            elif step.carrot is not None:
                err = (math.atan2(step.carrot[1] - ey, step.carrot[0] - ex) - eth + math.pi) % (2 * math.pi) - math.pi
                cmd = (0.25 if abs(err) < math.radians(45) else 0.0, max(-0.5, min(0.5, 1.2 * err)))
            else:
                cmd = (0.0, 0.0)
        queue.append(cmd)
        v, w = queue.popleft()
        nx, ny = x + v * math.cos(th) * dt, y + v * math.sin(th) * dt
        contact = obs.collided((nx, ny), things)
        if contact:                  # a bump: it does not go through, and it is counted once
            hits += not in_contact
        else:
            x, y = nx, ny
            ex, ey = ex + wheel_scale * v * math.cos(eth) * dt, ey + wheel_scale * v * math.sin(eth) * dt
        in_contact = contact
        th += w * dt
        eth += gyro_scale * w * dt
        t += dt
        if record and k % 10 == 0:
            trail.append((x, y, ex, ey))
    out = dict(seed=seed, done=m.done, explored=m.explored, shown=len(shown), t=round(t), hits=hits, stucks=m.stucks,
               distinct=len({s[2] for s in shown}), show_dist=[s[1] for s in shown],
               mark_err=[s[4] for s in shown], objects=len(m.objects()),
               pose_err=round(math.hypot(ex - x, ey - y), 2))
    return (out, m, np.array(trail)) if record else out


def plot(m, trail, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = m.map
    img = np.full(c.L.shape, 0.5)
    img[c.seen & (c.L < 0)] = 1.0
    img[c.occupied()] = 0.0
    r = c.p['resolution_m']
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(img, origin='lower', cmap='gray', vmin=0, vmax=1,
              extent=(c.ox, c.ox + c.w * r, c.oy, c.oy + c.h * r))
    for s in world():
        ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], color='tab:blue', lw=1, alpha=0.6)
    ax.plot(trail[:, 0], trail[:, 1], color='tab:green', lw=1, label='true path')
    ax.plot(trail[:, 2], trail[:, 3], color='tab:orange', lw=1, ls='--', label='odometry')
    ax.scatter(PLANTS[:, 0], PLANTS[:, 1], marker='o', s=60, facecolors='none', edgecolors='tab:green', label='plants (true)')
    for cls, ox, oy, visited, n in m.objects():
        ax.scatter([ox], [oy], marker='*', s=160, color='tab:red' if visited else 'tab:purple')
        ax.annotate(f'{cls}{" (shown)" if visited else ""}', (ox, oy), textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax.set_xlim(-4, 17); ax.set_ylim(-4, 14); ax.set_aspect('equal'); ax.legend(loc='lower right', fontsize=8)
    ax.set_title('map built by the mission (white free, black occupied, grey unseen); walls in blue for reference')
    fig.savefig(path, dpi=110, bbox_inches='tight')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--goals', type=int, default=3)
    ap.add_argument('--delay', type=float, default=1.0)
    ap.add_argument('--plot', default='')
    ap.add_argument('--count', action='store_true', help='drive past and count (no scans, no approach)')
    ap.add_argument('--reactive', action='store_true', help='the mission drives itself (reactive_drive)')
    a = ap.parse_args()
    for seed in range(a.seeds):
        if a.plot and seed == 0:
            out, m, trail = run(seed, a.goals, delay_s=a.delay, record=True, count=a.count, reactive=a.reactive)
            plot(m, trail, a.plot)
            print('\n'.join(m.notes[:60]))
        else:
            out = run(seed, a.goals, delay_s=a.delay, count=a.count, reactive=a.reactive)
        print(out)


if __name__ == '__main__':
    main()
