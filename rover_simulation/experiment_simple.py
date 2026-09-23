#!/usr/bin/env python3
"""One simple experiment, end to end, in pictures.

World: a building that OSM knows about and a planter wall that it does not.
  1. GLOBAL costmap: the building, rasterised like erc_static_map does (lethal =
     100). The global route is planned on it with erc_astar_planner_node's own
     astar_grid and reduced with checkpoint_controller_node's own shortcutting.
  2. The rover drives that route in controller_sim (carrot 1.5 m, side-step and
     local replanning on -- the field configuration). The planter wall sits on
     the global route.
  3. FREE SPACE: the perception profile at the moment the wall is first mapped,
     as rays in the world and as range vs bearing.
  4. LOCAL costmap at the first replan, with the detour it planned.
  5. LOCAL costmap at the end, every route, and the TRAJECTORY.

    source /opt/ros/jazzy/setup.bash && source ~/lsdc_ws/install/setup.bash
    ~/lsdc/erc-omni-vla/.venv/bin/python3 experiment_simple.py [--seed 0] [--out experiment_simple.png]
"""
import argparse
import importlib.util
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
from erc_inference import local_planner as lp  # noqa: E402

cs, obs = common.cs, common.obs
BRIDGE = os.path.normpath(os.path.join(common.HERE, '..', '..', 'Earth-rover-ros2-bridge'))
ASTAR = os.path.join(BRIDGE, 'erc_static_map', 'erc_static_map', 'erc_astar_planner_node.py')

START, GOAL = (0.0, 0.0), (26.0, 4.0)
BUILDING = (7.0, 13.0, -6.0, 2.5)          # xmin, xmax, ymin, ymax: in OSM
RES = 0.5                                   # global costmap resolution, m


def global_costmap():
    """Occupancy grid [row, col] over the area, building lethal, as erc_static_map writes it."""
    x0, y0, x1, y1 = -5.0, -10.0, 32.0, 12.0
    w, h = int((x1 - x0) / RES), int((y1 - y0) / RES)
    xs = x0 + (np.arange(w) + 0.5) * RES
    ys = y0 + (np.arange(h) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    bx0, bx1, by0, by1 = BUILDING
    occ = (X >= bx0) & (X <= bx1) & (Y >= by0) & (Y <= by1)
    return occ, (x0, y0)


def global_route(occ, origin):
    spec = importlib.util.spec_from_file_location('astar', ASTAR)
    astar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(astar)
    cell = lambda p: (int((p[0] - origin[0]) / RES), int((p[1] - origin[1]) / RES))  # noqa: E731
    path = astar.astar_grid(occ, cell(START), cell(GOAL))
    pts = [(origin[0] + (c + 0.5) * RES, origin[1] + (r + 0.5) * RES) for c, r in path]
    pts[0], pts[-1] = START, GOAL
    # the node's own line-of-sight reduction, on an OccupancyGrid like the service returns
    from nav_msgs.msg import OccupancyGrid
    g = OccupancyGrid()
    g.info.resolution, g.info.height, g.info.width = RES, occ.shape[0], occ.shape[1]
    g.info.origin.position.x, g.info.origin.position.y = origin
    g.data = (occ.astype(np.int8) * 100).ravel().tolist()
    node = cs.load_checkpoint_node()
    return node.CheckpointControllerNode._shortcut_path(pts, g, 15.0), np.array(pts)


def planter_on(route):
    """A 4 m wall across the route, 60% of the way along it, open past its left end."""
    r = np.asarray(route)
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(r, axis=0).T))])
    s = 0.62 * cum[-1]
    i = int(np.searchsorted(cum, s)) - 1
    a, b = r[i], r[i + 1]
    d = (b - a) / np.linalg.norm(b - a)
    p = a + d * (s - cum[i])
    n = np.array([-d[1], d[0]])                 # left of travel
    return obs.Segment(tuple(p - 3.0 * n), tuple(p + 0.8 * n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='experiment_simple.png')
    a = ap.parse_args()

    occ, origin = global_costmap()
    route, astar_cells = global_route(occ, origin)
    wall = planter_on(route)
    cs.SCENARIOS['experiment'] = cs.Scenario('experiment', (0.0, 0.0, 0.0), [list(route[1:])], [wall])

    # record what perception returned, where the rover truly was, and the map at the first replan
    profiles, snaps = [], {}
    real_profile = obs.profile

    def profile(x, y, th, world, rng=None, **kw):
        out = real_profile(x, y, th, world, rng, **kw)
        profiles.append((x, y, th, *out))
        return out
    cs.obs.profile = profile
    real_check = lp.LocalReplanner.check

    def check(self, t, x, y, pts, cum, s, extra_lethal=None):
        new, note = real_check(self, t, x, y, pts, cum, s, extra_lethal)
        if new is not None:
            snaps.setdefault('routes', []).append((t, np.asarray(new)))
            if 'first' not in snaps:
                snaps['first'] = (t, self.map.L.copy(), self.map.seen.copy(), (x, y), np.asarray(new))
        return new, note
    lp.LocalReplanner.check = check

    r, side = common.run('experiment', a.seed, 'lp+ss', 1.5, record=True)
    m = r.trace['planner'].map
    ext = [m.ox, m.ox + m.w * m.p['resolution_m'], m.oy, m.oy + m.h * m.p['resolution_m']]
    true = r.trace['true']
    print(f'hit={r.hit} completed={r.completed} t={r.time_s:.0f}s replans={r.replans} '
          f'side-step={side.state} min clearance={r.min_clearance_m:.2f} m')
    for note in r.trace['local_notes']:
        print(' ', *note)

    # the profile that saw most of the wall inside the mapped +-40 deg
    def n_hits(p):
        return int(((p[4] < p[5] - 1e-3) & (np.abs(p[3]) <= 40)).sum())
    best = max(profiles, key=n_hits)
    px, py, pth, pb, pf, pc = best

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    view = dict(xlim=(-2, 29), ylim=(-8, 9))

    def world(axx, title):
        axx.add_patch(plt.Rectangle((BUILDING[0], BUILDING[2]), BUILDING[1] - BUILDING[0],
                                    BUILDING[3] - BUILDING[2], color='0.35', label='building (in OSM)'))
        axx.plot([wall.a[0], wall.b[0]], [wall.a[1], wall.b[1]], 'k-', lw=4, label='planter (NOT in OSM)')
        axx.plot(*START, 'go', ms=8)
        axx.plot(*GOAL, 'r*', ms=14)
        axx.set_aspect('equal')
        axx.set(**view)
        axx.grid(alpha=.3)
        axx.set_title(title, fontsize=10)

    # (a) global costmap + global route
    a0 = ax[0, 0]
    a0.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.6,
              extent=[origin[0], origin[0] + occ.shape[1] * RES, origin[1], origin[1] + occ.shape[0] * RES])
    world(a0, '1. GLOBAL costmap (OSM) + A* route')
    a0.plot(astar_cells[:, 0], astar_cells[:, 1], '.', color='orange', ms=3, alpha=.5, label='A* cells')
    rt = np.array(route)
    a0.plot(rt[:, 0], rt[:, 1], 'o-', color='darkorange', lw=2, label='global route (shortcut)')
    a0.legend(fontsize=7, loc='lower right')

    # (b) free space, in the world
    a1 = ax[0, 1]
    world(a1, f'2. FREE SPACE (perception): rays, red = hit the wall')
    for b, f, c in zip(pb, pf, pc):
        ang = pth + math.radians(b)
        hit = f < c - 1e-3
        a1.plot([px, px + f * math.cos(ang)], [py, py + f * math.sin(ang)],
                '-', color='tab:red' if hit else 'tab:green', lw=1, alpha=.7)
        if hit:
            a1.plot(px + f * math.cos(ang), py + f * math.sin(ang), 'r.', ms=6)
    a1.plot(px, py, 'b^', ms=10, label='rover (true pose)')
    a1.set(xlim=(px - 4, px + 5), ylim=(py - 4.5, py + 4.5))
    a1.legend(fontsize=7, loc='lower right')

    # (c) the same profile, range vs bearing
    a2 = ax[0, 2]
    hit = pf < pc - 1e-3
    a2.bar(pb, pf, width=np.diff(pb).mean() * 0.9, color=np.where(hit, 'tab:red', 'tab:green'))
    a2.plot(pb, pc, 'k--', lw=1, label='what the sensor can see at all')
    a2.axvspan(-40, 40, color='tab:blue', alpha=.07, label='enters the local map (+-40 deg)')
    a2.axhline(1.2, color='tab:red', lw=.8, ls=':', label='side-step brake (1.2 m)')
    a2.set_xlabel('bearing (deg, + = left)')
    a2.set_ylabel('free distance (m)')
    a2.invert_xaxis()
    a2.legend(fontsize=7)
    a2.set_title('3. FREE SPACE profile (/erc/free_space): red = obstacle', fontsize=10)

    def local(axx, L, seen, title):
        axx.imshow(np.where(seen, L, np.nan), origin='lower', extent=ext, cmap='RdBu_r', vmin=-2, vmax=3.5)
        world(axx, title)
        axx.plot(rt[:, 0], rt[:, 1], '-', color='darkorange', lw=1.5, label='global route')

    # (d) local costmap at the first replan
    a3 = ax[1, 0]
    if 'first' in snaps:
        t1, L1, seen1, (x1, y1), new1 = snaps['first']
        local(a3, L1, seen1, f'4. LOCAL costmap at the first replan (t={t1:.0f} s)')
        a3.plot(new1[:, 0], new1[:, 1], 'g--', lw=2, label='first detour (wall only partly seen)')
        later = [n for t, n in snaps['routes'] if t > t1]
        if later:
            a3.plot(later[0][:, 0], later[0][:, 1], 'm--', lw=2,
                    label='next replan, more wall seen -> rejoins global route')
        a3.plot(x1, y1, 'b^', ms=10, label='rover (estimated)')
        a3.legend(fontsize=7, loc='lower right')
    else:
        a3.set_title('4. no replan happened', fontsize=10)

    # (e) local costmap at the end
    a4 = ax[1, 1]
    local(a4, m.L, m.seen, '5. LOCAL costmap at the end (red occupied, blue seen free)')
    a4.legend(fontsize=7, loc='lower right')

    # (f) trajectory over everything
    a5 = ax[1, 2]
    local(a5, m.L, m.seen, f'6. TRAJECTORY: hit={r.hit}, done={r.completed}, {r.time_s:.0f} s')
    for i, (t, n) in enumerate(snaps.get('routes', [])):
        a5.plot(n[:, 0], n[:, 1], '--', lw=1, label=f'local route t={t:.0f}s' if i < 5 else None)
    a5.plot(true[:, 2], true[:, 3], 'b-', lw=2, label='rover (true)')
    a5.legend(fontsize=7, loc='lower right')

    fig.suptitle('controller_sim: carrot 1.5 m + side-step + local replanning  |  '
                 'global = what OSM knows, local = what the camera saw', fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=85)
    print('->', a.out)


if __name__ == '__main__':
    main()
