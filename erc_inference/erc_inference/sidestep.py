"""Side-step around an obstacle: stop, turn ~60 deg in place, advance, resume.

Why this and not continuous steering. Every avoidance selector tried in
controller_sim (a gap selector, VFH+ additive, VFH+ with the carrot suppressed)
steered WHILE moving, and hit every obstacle in every scenario. The reason is
the plant: moving, the rover executes about 0.36 of the commanded yaw rate
(minimum radius ~2.8 m at 0.3 m/s); turning in place it executes 1.18
(mission_16sept, Mission17sept_wuhan 4.1). A manoeuvre built on turning in
place uses the one thing this rover turns well at.

States, all fed with what the node actually has -- ESTIMATED heading and
position, and the /erc/free_space profile -- never the truth:

  FOLLOW   pass the controller's command through. Something nearer than
           stop_distance_m straight ahead for confirm_ticks -> BRAKE.
  BRAKE    v = w = 0 for brake_s, then pick a side: the one the route is
           on (the carrot's bearing) if the profile shows it open, else the
           open one. Neither side open -> GIVE_UP.
  TURN     How far: parallel to the obstacle's face plus turn_away_deg, from
           a line fitted to the profile's near points, within turn_min_deg ..
           turn_max_deg (turn_deg if no line). A fixed 60 deg against a
           perpendicular kerb points the advance INTO it (1.3 m at 60 deg
           closes 0.65 m); against the 22-sept hedge, nearly parallel to the
           route, 60 is plenty and 90 wastes the advance.
           w = +-turn_w in place until the heading, plus what the rover will
           still turn on its own, reaches turn_deg. "On its own" is the
           measured turn rate x lead_s: the command acts ~1.3 s late and the
           compass reports ~0.7 s late, so the rover keeps turning ~2 s after
           the command is cut. Rate, not a fixed angle: it adapts to surfaces
           where the rover turns slower.
  SETTLE   v = w = 0 for settle_s, long enough for that residual turn to
           land and for the camera to look along the new heading.

Field, 2026-09-22 (mission_carrot_sidestep*, 22 turns, 90 deg and a fixed
25 deg lead): the compass started moving 2.7 s (p50) after the command, turned
at 18.6 deg/s, kept turning 3.2 s and 37 deg (p90 50) after the command was
cut, and ended at 99-126 deg. controller_sim reproduces it (103 deg at
ADVANCE, field ~100). Hence 60 deg, lead_s 2.4 (measured 2.0 = 37 / 18.6), settle_s 3.0.
  ADVANCE  straight ahead (now sideways to the route) for advance_m, as long as
           the profile says it is clear. Blocked -> GIVE_UP.
           Blocked before it has moved (the corner of what it just braked for,
           stopped closer than planned by the delay): back to TURN for
           extra_turn_deg more on the same side, max_extra_turns times.
  -> FOLLOW. The carrot then pulls the rover back onto the route; the
           controller must be reset (step() says when) because its delay
           compensation recorded turns it never commanded.
  GIVE_UP  v = w = 0 and stay. An operator takes over.

Repeated side-steps keep the SAME side within retry_window_s, so a long wall
is walked along in steps instead of alternating sides; more than max_attempts
in that window -> GIVE_UP.

No ROS, no torch: omnivla_erc_inference's rule is that the node and
controller_sim run the same code, and this is imported by both.
"""
import math

import numpy as np

FOLLOW, BRAKE, TURN, SETTLE, ADVANCE, GIVE_UP = (
    'follow', 'brake', 'turn', 'settle', 'advance', 'give-up')

DEFAULTS = dict(
    stop_distance_m=1.2,     # covers the 1.3 s loop delay; 0.8 hit in simulation
    sector_deg=20.0,         # "straight ahead"
    confirm_ticks=2,
    brake_s=1.0,
    turn_w=0.3,              # rad/s, the same in-place rate GoalTurn commands
    turn_deg=60.0,           # when the wall's direction cannot be fitted
    turn_min_deg=45.0,       # else: parallel to the wall + turn_away_deg,
    turn_max_deg=110.0,      #   within these bounds. 90 lost kerb runs the old
                             #   overshoot to ~103 had cleared (sim, carrot 1.5 m)
    turn_away_deg=15.0,
    wall_fit_range_m=1.4,    # profile points nearer than this are the wall; the
                             # periphery reads ~1.7 m with nothing there (TAREA5)
    wall_fit_min_points=4,
    extra_turn_deg=30.0,     # ADVANCE blocked before moving: turn this much more,
    max_extra_turns=2,       #   at most this many times, then give up
    lead_s=2.4,              # measured 2.0 (37 deg at 18.6 deg/s); 2.4 lands 61-75
                             # in controller_sim -- ~60 is the floor at turn_w 0.3
    lead_min_deg=8.0,        # before the rate estimate has anything to go on
    settle_s=3.0,            # field: the turn ran on 3.2 s (p50) after the cut
    goal_side_min_deg=10.0,  # carrot this far off the heading decides the side
    advance_v=0.25,          # min_linear_vel: slower does not move the rover
    advance_m=1.3,
    side_open_m=0.6,         # best direction on a side must reach this
    retry_window_s=60.0,
    max_attempts=4,
)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SideStep:
    def __init__(self, **kw):
        self.p = dict(DEFAULTS)
        self.p.update(kw)
        self.reset()

    def reset(self):
        self.state = FOLLOW
        self._n = 0
        self._t0 = None
        self._h0 = None
        self._xy0 = None
        self._side = 0
        self._attempts = []          # times of recent side-steps
        self._pref_side = 0
        self._hist = []              # (t, heading) while turning, for the rate
        self._target = self.p['turn_deg']
        self._extra = 0

    # -- profile helpers --------------------------------------------------
    def _sector_min(self, bearings, free, lo, hi):
        b, f = np.asarray(bearings, float), np.asarray(free, float)
        k = (b >= lo) & (b <= hi) & np.isfinite(f)
        return float(f[k].min()) if k.any() else float('inf')

    def _ahead(self, bearings, free):
        s = self.p['sector_deg']
        return self._sector_min(bearings, free, -s, s)

    def _pick_side(self, bearings, free, now, goal_bearing=None):
        """+1 left, -1 right, 0 none.

        Order: the route's side (goal_bearing, left-positive rad) when it is
        open; then the side used in the last retry_window_s, so a wall is
        walked along instead of alternating; then the more open side. In the
        field the old rule (free space only) turned right 18 times in 23.

        By the BEST open direction on each side, not the median. A first version
        used the median and gave up on the first kerb approach at L=0.81 R=0.87:
        the median mixed bins that hit the kerb with periphery bins the sensor
        only sees to ~1.7 m (TAREA5 6c.4), so neither side ever looked open.
        A passage shows as SOME direction being open, not most of them.

        And this is only a preference. The camera sees +-60 deg; the 90 deg
        direction the rover is about to face cannot be seen before turning.
        The real safety check is ADVANCE's, made after the turn. So the bar here
        is low (side_open_m) and giving up needs both sides clearly shut.
        """
        b, f = np.asarray(bearings, float), np.asarray(free, float)
        s = self.p['sector_deg']
        L = f[b > s]
        R = f[b < -s]
        left = float(L.max()) if L.size else 0.0
        right = float(R.max()) if R.size else 0.0
        need = self.p['side_open_m']
        if max(left, right) < need:
            return 0, left, right
        if (goal_bearing is not None
                and abs(math.degrees(goal_bearing)) >= self.p['goal_side_min_deg']):
            g = 1 if goal_bearing > 0 else -1
            if (left if g > 0 else right) >= need:
                return g, left, right
        recent = [t for t in self._attempts if now - t < self.p['retry_window_s']]
        if recent and self._pref_side:
            if (left if self._pref_side > 0 else right) >= need:
                return self._pref_side, left, right
        if max(left, right) < need:
            return 0, left, right
        if abs(left - right) < 0.05:
            # tie on the best direction: prefer the side that is open more widely
            left_med = float(np.median(L)) if L.size else 0.0
            right_med = float(np.median(R)) if R.size else 0.0
            return (1 if left_med >= right_med else -1), left, right
        return (1 if left > right else -1), left, right

    def _turn_target(self, bearings, free, side):
        """Degrees to turn towards `side` to run parallel to the obstacle + turn_away_deg."""
        p = self.p
        b, f = np.radians(np.asarray(bearings, float)), np.asarray(free, float)
        k = np.isfinite(f) & (f < p['wall_fit_range_m'])
        if k.sum() < p['wall_fit_min_points']:
            return p['turn_deg'], None
        pts = np.stack([f[k] * np.cos(b[k]), f[k] * np.sin(b[k])], axis=1)
        pts -= pts.mean(axis=0)
        # principal direction of the near points = the face's direction
        _u, _s, vt = np.linalg.svd(pts, full_matrices=False)
        phi = math.degrees(math.atan2(vt[0, 1], vt[0, 0])) % 180.0   # 0..180, left-positive
        parallel = phi if side > 0 else 180.0 - phi
        target = min(max(parallel + p['turn_away_deg'], p['turn_min_deg']), p['turn_max_deg'])
        return target, phi

    # -- the machine -------------------------------------------------------
    def _rate_dps(self, now):
        """Turn rate over the last second of the heading, deg/s, in the turn's direction."""
        h = [(t, x) for t, x in self._hist if now - t <= 1.0]
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.5:
            return 0.0
        d = math.degrees(_wrap(h[-1][1] - h[0][1])) * self._side
        return max(0.0, d / (h[-1][0] - h[0][0]))

    def step(self, now, heading, xy, bearings, free, v_cmd, w_cmd, goal_bearing=None):
        """-> (v, w, note, resumed). `resumed` True means reset the controller now.

        heading: yaw in rad, counter-clockwise positive (any zero). goal_bearing:
        the carrot's bearing from the rover, rad, left-positive, or None.
        """
        p = self.p
        ahead = self._ahead(bearings, free)

        if self.state == FOLLOW:
            if v_cmd > 0.0 and ahead < p['stop_distance_m']:
                self._n += 1
            else:
                self._n = 0
            if self._n >= p['confirm_ticks']:
                recent = [t for t in self._attempts if now - t < p['retry_window_s']]
                if len(recent) >= p['max_attempts']:
                    self.state = GIVE_UP
                    return 0.0, 0.0, f'[side-step] give up: {len(recent)} attempts', False
                self.state, self._t0, self._n = BRAKE, now, 0
                return 0.0, 0.0, f'[side-step] brake ahead={ahead:.2f}m', False
            return v_cmd, w_cmd, '', False

        if self.state == BRAKE:
            if now - self._t0 < p['brake_s']:
                return 0.0, 0.0, '[side-step] brake', False
            side, left, right = self._pick_side(bearings, free, now, goal_bearing)
            if side == 0:
                self.state = GIVE_UP
                return 0.0, 0.0, f'[side-step] give up: L={left:.2f} R={right:.2f}', False
            self._side, self._pref_side = side, side
            self._attempts.append(now)
            self.state, self._h0, self._t0 = TURN, heading, now
            self._hist = [(now, heading)]
            self._extra = 0
            self._target, phi = self._turn_target(bearings, free, side)
            wall = 'none' if phi is None else f'{phi:.0f}deg'
            return 0.0, side * p['turn_w'], (f'[side-step] turn {"left" if side > 0 else "right"}'
                                            f' L={left:.2f} R={right:.2f} wall={wall}'
                                            f' target={self._target:.0f}deg'), False

        if self.state == TURN:
            turned = math.degrees(_wrap(heading - self._h0)) * self._side
            self._hist.append((now, heading))
            rate = self._rate_dps(now)
            lead = max(p['lead_min_deg'], rate * p['lead_s'])
            if turned + lead >= self._target:
                self.state, self._t0 = SETTLE, now
                return 0.0, 0.0, (f'[side-step] settle turned={turned:.0f}deg '
                                  f'rate={rate:.0f}dps lead={lead:.0f}deg'), False
            # field: 2.7 s (p90 4.2) before the compass even moves
            if now - self._t0 > 6.0 + 2.0 * self._target / math.degrees(p['turn_w']):
                self.state = GIVE_UP
                return 0.0, 0.0, f'[side-step] give up: turn stalled at {turned:.0f}deg', False
            return 0.0, self._side * p['turn_w'], f'[side-step] turn {turned:.0f}deg', False

        if self.state == SETTLE:
            if now - self._t0 < p['settle_s']:
                return 0.0, 0.0, '[side-step] settle', False
            self.state, self._xy0, self._t0 = ADVANCE, tuple(xy), now
            return 0.0, 0.0, '[side-step] advance', False

        if self.state == ADVANCE:
            moved = math.hypot(xy[0] - self._xy0[0], xy[1] - self._xy0[1])
            if ahead < p['stop_distance_m'] * 0.75 and moved < 0.2 and self._extra < p['max_extra_turns']:
                self._extra += 1
                self.state, self._h0, self._t0 = TURN, heading, now
                self._hist = [(now, heading)]
                self._target = p['extra_turn_deg']
                return 0.0, self._side * p['turn_w'], (f'[side-step] still blocked at {ahead:.2f}m: '
                                                      f'turn {self._target:.0f}deg more'), False
            if ahead < p['stop_distance_m'] * 0.75:
                self.state = GIVE_UP
                return 0.0, 0.0, f'[side-step] give up: side blocked at {ahead:.2f}m', False
            if moved >= p['advance_m'] or now - self._t0 > 4.0 * p['advance_m'] / p['advance_v']:
                self.state = FOLLOW
                return 0.0, 0.0, f'[side-step] resume after {moved:.2f}m', True
            return p['advance_v'], 0.0, f'[side-step] advance {moved:.2f}m', False

        return 0.0, 0.0, '[side-step] give up (holding)', False
