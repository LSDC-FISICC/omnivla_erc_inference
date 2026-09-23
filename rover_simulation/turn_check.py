#!/usr/bin/env python3
"""How far SideStep really turns in simulation, measured on the estimated heading
as the field analysis measured it on /erc/heading_deg.

Field reference (mission_carrot_sidestep*, 90 deg target, fixed 25 deg lead):
~100 deg when ADVANCE starts. This simulator reproduced 103 with those settings.

    python3 turn_check.py                       # current sidestep.DEFAULTS
    python3 turn_check.py --ss '{"lead_s": 2.8}' --latency 0.7
"""
import argparse
import json
import math
from dataclasses import replace

import numpy as np

import common


class Recorder(common.SideStep):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.log = []

    def step(self, now, heading, *a, **k):
        out = super().step(now, heading, *a, **k)
        self.log.append((now, self.state, heading))
        return out


def turns(log):
    """(turn at ADVANCE, max turn before FOLLOW) per manoeuvre, degrees."""
    out = []
    for i in range(1, len(log)):
        if log[i][1] == 'turn' and log[i - 1][1] not in ('turn', 'settle', 'advance'):
            h0 = log[i][2]
            rel = [(s, abs(math.degrees(common.cs.mc.clip_angle(h - h0)))) for _t, s, h in log[i:]]
            end = next((k for k, (s, _d) in enumerate(rel) if s == 'follow'), len(rel) - 1)
            adv = next((d for s, d in rel if s == 'advance'), float('nan'))
            out.append((adv, max(d for _s, d in rel[:end + 1])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ss', default='{}')
    ap.add_argument('--latency', type=float, default=0.25, help='RoverModel.sensor_latency_s')
    ap.add_argument('--seeds', type=int, default=4)
    a = ap.parse_args()
    model = replace(common.cs.RoverModel(), sensor_latency_s=a.latency)
    rows = []
    for scenario in ('kerb', 'post', 'chicane'):
        for seed in range(a.seeds):
            rec = Recorder(**json.loads(a.ss))
            common.run(scenario, seed, 'ss', model=model, override=rec)
            rows += turns(rec.log)
    r = np.array(rows)
    print(f'{len(r)} turns: at ADVANCE p50 {np.nanmedian(r[:, 0]):.0f} deg '
          f'[p10 {np.nanpercentile(r[:, 0], 10):.0f}, p90 {np.nanpercentile(r[:, 0], 90):.0f}]; '
          f'max before FOLLOW p50 {np.median(r[:, 1]):.0f}')


if __name__ == '__main__':
    main()
