"""Side-step around an obstacle: stop, turn 90 deg in place, advance, resume.

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
  BRAKE    v = w = 0 for brake_s, then pick the side with more free space in
           the profile. Neither side open -> GIVE_UP.
  TURN     w = +-turn_w in place until the estimated heading has moved
           turn_deg - lead_deg. The lead is the loop delay: a command keeps
           acting ~1.3 s after it stops, so the rover finishes the turn on its own.
  SETTLE   v = w = 0 for settle_s, long enough for that residual turn to
           land and for the camera to look along the new heading.
  ADVANCE  straight ahead (now sideways to the route) for advance_m, as long as
           the profile says it is clear. Blocked -> GIVE_UP.
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
    turn_deg=90.0,
    lead_deg=25.0,           # ~20 deg/s real in-place rate x the 1.3 s delay
    settle_s=1.5,
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

    # -- profile helpers --------------------------------------------------
    def _sector_min(self, bearings, free, lo, hi):
        b, f = np.asarray(bearings, float), np.asarray(free, float)
        k = (b >= lo) & (b <= hi) & np.isfinite(f)
        return float(f[k].min()) if k.any() else float('inf')

    def _ahead(self, bearings, free):
        s = self.p['sector_deg']
        return self._sector_min(bearings, free, -s, s)

    def _pick_side(self, bearings, free, now):
        """+1 left, -1 right, 0 none. Keeps the recent side while walking a wall.

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

    # -- the machine -------------------------------------------------------
    def step(self, now, heading, xy, bearings, free, v_cmd, w_cmd):
        """-> (v, w, note, resumed). `resumed` True means reset the controller now."""
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
            side, left, right = self._pick_side(bearings, free, now)
            if side == 0:
                self.state = GIVE_UP
                return 0.0, 0.0, f'[side-step] give up: L={left:.2f} R={right:.2f}', False
            self._side, self._pref_side = side, side
            self._attempts.append(now)
            self.state, self._h0, self._t0 = TURN, heading, now
            return 0.0, side * p['turn_w'], (f'[side-step] turn {"left" if side > 0 else "right"}'
                                            f' L={left:.2f} R={right:.2f}'), False

        if self.state == TURN:
            turned = math.degrees(_wrap(heading - self._h0)) * self._side
            if turned >= p['turn_deg'] - p['lead_deg']:
                self.state, self._t0 = SETTLE, now
                return 0.0, 0.0, f'[side-step] settle turned={turned:.0f}deg', False
            if now - self._t0 > 3.0 * p['turn_deg'] / math.degrees(p['turn_w']):
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
            if ahead < p['stop_distance_m'] * 0.75:
                self.state = GIVE_UP
                return 0.0, 0.0, f'[side-step] give up: side blocked at {ahead:.2f}m', False
            if moved >= p['advance_m'] or now - self._t0 > 4.0 * p['advance_m'] / p['advance_v']:
                self.state = FOLLOW
                return 0.0, 0.0, f'[side-step] resume after {moved:.2f}m', True
            return p['advance_v'], 0.0, f'[side-step] advance {moved:.2f}m', False

        return 0.0, 0.0, '[side-step] give up (holding)', False
