#!/usr/bin/env python3
"""One simulated mission to a PNG: the local map as the rover built it (from its
ESTIMATED pose), the real obstacles, the true path and every local replan.

    python3 plot_run.py planter 0                 # lp+ss, carrot 1.5 m
    python3 plot_run.py hedge 3 --config ss --out /tmp/hedge_ss.png
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
from erc_inference import local_planner as lp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scenario')
    ap.add_argument('seed', type=int)
    ap.add_argument('--config', default='lp+ss')
    ap.add_argument('--carrot', type=float, default=1.5)
    ap.add_argument('--ss', default='{}')
    ap.add_argument('--lp', default='{}')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    routes = []
    original = lp.LocalReplanner.check

    def check(self, t, x, y, pts, cum, s, extra_lethal=None):
        new, note = original(self, t, x, y, pts, cum, s, extra_lethal)
        if new is not None:
            routes.append((t, np.asarray(new)))
        return new, note
    lp.LocalReplanner.check = check

    r, side = common.run(a.scenario, a.seed, a.config, a.carrot, json.loads(a.ss), json.loads(a.lp), record=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    planner = r.trace.get('planner')
    if planner is not None:
        m = planner.map
        res = m.p['resolution_m']
        ax.imshow(np.where(m.seen, m.L, np.nan), origin='lower', cmap='RdBu_r', vmin=-2, vmax=3.5,
                  extent=[m.ox, m.ox + m.w * res, m.oy, m.oy + m.h * res], alpha=.8)
    for o in common.cs.SCENARIOS[a.scenario].obstacles:
        if hasattr(o, 'a'):
            ax.plot([o.a[0], o.b[0]], [o.a[1], o.b[1]], 'k-', lw=3)
        else:
            ax.add_patch(plt.Circle(o.c, o.r, color='k'))
    true = r.trace['true']
    ax.plot(true[:, 2], true[:, 3], 'g-', lw=1.5, label='true path')
    for i, (t, n) in enumerate(routes):
        ax.plot(n[:, 0], n[:, 1], '--', lw=1, label=f'replan t={t:.0f}s' if i < 10 else None)
    ax.set_aspect('equal')
    ax.grid(alpha=.3)
    ax.legend(fontsize=6, loc='lower right')
    ax.set_title(f'{a.scenario} seed {a.seed} {a.config} carrot {a.carrot} m: hit={r.hit} '
                 f'done={r.completed} t={r.time_s:.0f}s replans={r.replans} '
                 f'side-step={side.state if side else "-"}', fontsize=9)
    out = a.out or f'{a.scenario}_{a.seed}_{a.config}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    for note in r.trace['local_notes']:
        print(*note)
    print('->', out)


if __name__ == '__main__':
    main()
