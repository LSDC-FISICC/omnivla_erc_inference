#!/usr/bin/env python3
"""Obstacles for controller_sim, and the free-space profile a rover would see.

controller_sim models route geometry, the plant and localisation, but the world
is empty: every scenario is a clear path. That was fine while the only thing
being validated was tracking the A* route. It is not fine for obstacle
avoidance, whose real failure mode is not picking a bad bearing in one frame --
it is OSCILLATING, going back and forth between two openings or in and out of a
hazard, and that only appears in closed loop.

Two things live here:

1. The world: segments (kerbs, walls, steps) and discs (posts, another rover).
   A kerb is a segment because that is its shape; approximating it with discs
   puts phantom gaps between them.

2. `profile()`: what the perception stack would report from a given pose --
   distance to the first thing standing proud of the ground, per bearing bin.
   It is deliberately NOT a perfect ray cast. The measured behaviour of the
   real profile (erc_perception, TAREA5 6c.4, 30 frames of mission_17sept_wuhan_2
   with driven ground as truth) is reproduced:

     - it saw >=1 m clear in 97% of frames where >=1 m was verified clear, so
       the centre is close to honest but not perfect
     - the periphery is much worse: median free distance 1.70 m at -58 deg and
       1.69 m at +58 deg, against 3.00 m (the cap) between -14 and +29 deg

   Simulating a perfect sensor would validate a selector that cannot exist.
"""
import math

import numpy as np

FOV_DEG = 60.0
MAX_RANGE_M = 3.0
N_BINS = 25
FOOTPRINT_W = 0.25          # MINI+ track, spec sheet


class Segment:
    """A wall, kerb or step: everything on the line is untraversable."""

    def __init__(self, a, b):
        self.a = np.asarray(a, float)
        self.b = np.asarray(b, float)

    def distance_to(self, p):
        p = np.asarray(p, float)
        ab = self.b - self.a
        denom = float(ab @ ab)
        t = 0.0 if denom < 1e-12 else float(np.clip((p - self.a) @ ab / denom, 0.0, 1.0))
        return float(np.linalg.norm(p - (self.a + t * ab)))

    def ray_hit(self, o, d, max_t):
        """Distance along unit ray `d` from `o` to this segment, or inf."""
        v1 = o - self.a
        v2 = self.b - self.a
        v3 = np.array([-d[1], d[0]])
        denom = float(v2 @ v3)
        if abs(denom) < 1e-12:
            return math.inf
        # 2-D cross by hand: numpy 2 removed the 2-vector form of np.cross
        t1 = (v2[0] * v1[1] - v2[1] * v1[0]) / denom     # along the ray
        t2 = float(v1 @ v3) / denom              # along the segment
        if t1 < 0.0 or not (0.0 <= t2 <= 1.0) or t1 > max_t:
            return math.inf
        return t1


class Disc:
    """A post, a bin, another rover."""

    def __init__(self, centre, radius):
        self.c = np.asarray(centre, float)
        self.r = float(radius)

    def distance_to(self, p):
        return max(0.0, float(np.linalg.norm(np.asarray(p, float) - self.c)) - self.r)

    def ray_hit(self, o, d, max_t):
        f = o - self.c
        b = float(f @ d)
        c = float(f @ f) - self.r ** 2
        disc = b * b - c
        if disc < 0.0:
            return math.inf
        s = math.sqrt(disc)
        for t in (-b - s, -b + s):
            if 0.0 <= t <= max_t:
                return t
        return math.inf


def clearance(pose_xy, obstacles):
    """Distance from the rover centre to the nearest obstacle."""
    if not obstacles:
        return math.inf
    return min(o.distance_to(pose_xy) for o in obstacles)


def collided(pose_xy, obstacles, half_width=FOOTPRINT_W / 2):
    return clearance(pose_xy, obstacles) <= half_width


# measured periphery falloff, TAREA5 6c.4: median free distance by bearing
_MEASURED_BEARING = np.array([-58, -43, -29, -14, 0, 14, 29, 43, 58], float)
_MEASURED_RANGE = np.array([1.70, 2.19, 2.84, 3.00, 3.00, 3.00, 3.00, 2.40, 1.69])


def _range_cap(bearings_deg):
    """How far the real profile actually sees, per bearing."""
    return np.interp(bearings_deg, _MEASURED_BEARING, _MEASURED_RANGE)


def profile(x, y, theta, obstacles, rng=None, n_bins=N_BINS, fov_deg=FOV_DEG,
            max_range=MAX_RANGE_M, miss_rate=0.03, noise_m=0.08):
    """Free distance per bearing bin, as the perception stack would report it.

    `miss_rate` is the 3% of the measured check where >=1 m of verified clear
    ground was NOT reported clear; here it is applied the dangerous way round
    as well -- a bin can also miss an obstacle -- because the measurement only
    bounds one of the two errors and assuming the other is zero would be
    wishful.
    """
    rng = rng or np.random.default_rng(0)
    o = np.array([x, y], float)
    edges = np.linspace(-fov_deg, fov_deg, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    caps = _range_cap(centres)

    free = np.empty(n_bins)
    for i, b in enumerate(centres):
        ang = theta + math.radians(b)
        d = np.array([math.cos(ang), math.sin(ang)])
        hit = min((ob.ray_hit(o, d, max_range) for ob in obstacles), default=math.inf)
        hit = min(hit, caps[i])                       # cannot see past its own range
        if math.isfinite(hit) and hit < caps[i] - 1e-6:
            if rng.random() < miss_rate:
                hit = caps[i]                          # missed a real obstacle
            else:
                hit = max(0.05, hit + rng.normal(0.0, noise_m))
        free[i] = min(hit, max_range)
    # `caps` is what the sensor can see in that direction at all. Reporting it
    # separately lets a consumer tell 'nothing there' from 'cannot tell',
    # which matters because the measured periphery caps at ~1.7 m and would
    # otherwise read as a permanent obstacle at the edges of the view.
    return centres, free, caps


def selftest(verbose=True):
    ok = True
    rng = np.random.default_rng(0)

    # a wall straight ahead at 1.5 m, spanning the whole view
    wall = Segment((1.5, -5.0), (1.5, 5.0))
    b, f, _c = profile(0.0, 0.0, 0.0, [wall], rng, noise_m=0.0, miss_rate=0.0)
    centre = f[len(f) // 2]
    ok &= abs(centre - 1.5) < 0.05
    if verbose:
        print(f'wall at 1.5 m ahead        -> centre reads {centre:.2f} m  '
              f'{"PASS" if abs(centre-1.5) < 0.05 else "FAIL"}')

    # empty world: every bin reads its own range cap, not the global max
    b, f, _c = profile(0.0, 0.0, 0.0, [], rng, noise_m=0.0, miss_rate=0.0)
    edge_ok = f[0] < 2.0 and f[len(f) // 2] > 2.9
    ok &= edge_ok
    if verbose:
        print(f'empty world                -> centre {f[len(f)//2]:.2f} m, '
              f'edge {f[0]:.2f} m  {"PASS" if edge_ok else "FAIL"} '
              f'(periphery is measured to see less)')

    # a disc off to the left only
    disc = Disc((1.0, 1.0), 0.3)
    b, f, _c = profile(0.0, 0.0, 0.0, [disc], rng, noise_m=0.0, miss_rate=0.0)
    left = f[b > 30].min()
    right = f[b < -30].min()
    side_ok = left < 1.6 and right > 1.5
    ok &= side_ok
    if verbose:
        print(f'disc at +45 deg            -> left {left:.2f} m, right {right:.2f} m  '
              f'{"PASS" if side_ok else "FAIL"}')

    # collision uses the footprint, not the centre point
    ok &= collided((1.5 - 0.1, 0.0), [wall]) and not collided((1.0, 0.0), [wall])
    if verbose:
        print(f'footprint collision        -> {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if selftest() else 1)
