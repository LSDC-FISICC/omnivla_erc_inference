#!/usr/bin/env python3
"""VFH+ over the free-space profile.

Chosen over virtual potential fields for one concrete reason: kerbs, walls and
steps are EXTENDED surfaces with a gap to one side, and a repulsion vector from
a surface perpendicular to the route points straight back along it, cancelling
the attraction to the carrot. That is the textbook local minimum, and it is the
most common obstacle shape in this application. VFH does not sum forces -- it
looks for valleys in the polar histogram and steers into one.

The first attempt at a selector (avoidance.GapSelector) hit every obstacle in
all three scenarios. It was missing the two things that ARE VFH:

  * enlargement -- each obstacle is widened by the robot half-width plus a
    margin, so a valley that survives is one the rover actually fits through.
    Without it the selector picks the "least blocked" bearing, which can be a
    direction it cannot pass.
  * valleys, not bins -- a candidate has to be a contiguous free sector wide
    enough to drive into, not just a cheap bin with blocked neighbours.

Convention: the profile bearings and the carrot bearing are both in the body
frame (0 = straight ahead). `__call__` returns a DEVIATION to add to the carrot
bearing, because that is the slot MotionController exposes.
"""
import math

import numpy as np

FOOTPRINT_W = 0.25          # MINI+ track, spec sheet
SAFETY_M = 0.12


class VFHPlus:
    def __init__(self, max_range_m=3.0, robot_half_w=FOOTPRINT_W / 2,
                 safety_m=SAFETY_M,
                 max_dev_deg=30.0, mu_target=5.0, mu_heading=2.0, mu_prev=2.0,
                 min_valley_deg=None, d_block=2.2, d_clear=2.7,
                 blind_ticks=9, suppress_carrot=False):
        self.max_range = max_range_m
        self.clear_w = robot_half_w + safety_m
        # Shut below d_block, open above d_clear. d_block is set by the loop
        # delay, not by taste: 1.3 s at 0.3 m/s is 0.4 m travelled before a
        # command bites, so a threshold near the 1.5 m carrot would commit
        # the rover before it could turn.
        self.d_block, self.d_clear = d_block, d_clear
        # ~3 s at 3 Hz: long enough to carry a manoeuvre through a pinch,
        # short enough that a genuinely walled-in rover stops insisting.
        self.blind_ticks = blind_ticks
        self.suppress_carrot = suppress_carrot
        self.max_dev = math.radians(max_dev_deg)
        self.mu = (mu_target, mu_heading, mu_prev)
        # A valley only has to be non-empty. The enlargement already widened
        # every hazard by asin(clear_w / d), which IS the rover's width scaled
        # to distance, so anything still open is wide enough to drive into.
        # Requiring a further fixed width charged the same margin twice: as the
        # rover closed on the 16-sept kerb the enlargement grew (asin blows up
        # near the obstacle), the surviving valley narrowed 34 -> 29 -> 24 deg,
        # and at 28.4 deg VFH declared a gap it could actually fit through
        # impassable and gave up two metres out.
        self.min_valley = float(min_valley_deg or 0.0)
        self._binary = None
        self._prev = 0.0
        self._blind = 0

    # -- histogram ---------------------------------------------------------
    def _blocked(self, bearings_deg, free_m, caps=None):
        """Which bearings are shut, after enlarging by the rover's width.

        One mechanism, not two: an earlier version ramped a "density" over the
        range and then thresholded that, so the reaction distance was set twice
        in different units and a wall at 2.0 m landed at 0.43 against a 0.45
        threshold -- invisible for no reason anyone could read off the code.

        A bin is shut on POSITIVE evidence only: nearer than d_block, and
        nearer than what that direction can see at all. The measured profile
        caps at ~1.7 m in the periphery (TAREA5 6c.4), and without the second
        test those bins would read as a permanent wall at the edge of the view.
        """
        d = np.clip(np.asarray(free_m, float), 0.05, self.max_range)
        ceil = (np.full_like(d, self.max_range) if caps is None
                else np.clip(np.asarray(caps, float), 0.05, self.max_range))
        seen = d < ceil - 0.15

        if self._binary is None or len(self._binary) != len(d):
            self._binary = np.zeros(len(d), bool)
        # hysteresis on distance: shut below d_block, open above d_clear, and
        # whatever it was in between. Keeps a bin from flickering at the edge.
        shut = np.where(seen & (d < self.d_block), True,
                        np.where(d > self.d_clear, False, self._binary))

        # enlargement: a hazard at distance d shuts +-asin(clear_w/d) around
        # itself, so a valley that survives is one the rover actually fits in.
        b = np.asarray(bearings_deg, float)
        gamma = np.degrees(np.arcsin(np.clip(self.clear_w / d, 0.0, 1.0)))
        out = np.zeros(len(b), bool)
        for bi, gi, sh in zip(b, gamma, shut):
            if sh:
                out |= np.abs(b - bi) <= gi
        self._binary = out
        return out

    # -- valleys -----------------------------------------------------------
    @staticmethod
    def _valleys(blocked):
        out, start = [], None
        for i, b in enumerate(blocked):
            if not b and start is None:
                start = i
            elif b and start is not None:
                out.append((start, i - 1)); start = None
        if start is not None:
            out.append((start, len(blocked) - 1))
        return out

    def _candidates(self, bearings_deg, blocked, target_deg=None):
        b = np.asarray(bearings_deg, float)
        step = float(np.mean(np.diff(b))) if len(b) > 1 else 1.0
        half = max(self.min_valley, step) / 2.0
        cands = []
        for lo, hi in self._valleys(blocked):
            width = (hi - lo + 1) * step
            # one whole bin, not half: at half a bin a single noise-opened
            # cell counted as a passage and the selector invented a gap in a
            # solid wall (traced: 'wall at 1.5 m' returned -30 deg).
            if width < max(self.min_valley, step * 1.5):
                continue          # empty after enlargement: genuinely no room
            left, right = b[lo], b[hi]
            if width <= self.min_valley * 1.5:
                cands.append(0.5 * (left + right))          # narrow: centre it
            else:
                # wide: hug each edge at a safe offset, and -- the part the
                # first version was missing -- the target itself when it lies
                # inside. Without it the selector always preferred the valley's
                # middle and quietly threw the carrot away.
                cands += [left + half, right - half]
                if target_deg is not None and left + half <= target_deg <= right - half:
                    cands.append(float(target_deg))
                else:
                    cands.append(0.5 * (left + right))
        return cands

    # -- selection ---------------------------------------------------------
    def __call__(self, bearings_deg, free_m, carrot_bearing, caps=None):
        """-> deviation to add to the carrot bearing.

        `suppress_carrot` returns instead a deviation that CANCELS the carrot
        and substitutes the chosen bearing, for as long as anything is shut.
        The additive form is what the controller exposes today, but tracing the
        kerb showed the carrot growing against the manoeuvre as the rover
        displaced, so the two fight and the deviation is spent undoing the
        carrot rather than clearing the hazard.
        """
        blocked = self._blocked(bearings_deg, free_m, caps)
        target = math.degrees(carrot_bearing)
        cands = self._candidates(bearings_deg, blocked, target)
        if not cands:
            # Nothing passable in view. Returning zero hands the rover back to
            # the carrot -- which points at the obstacle, because that is why
            # the route needed leaving. Traced on the kerb scenario: VFH asked
            # for +30 deg for ten ticks, fell silent as the valley pinched, the
            # carrot pulled it straight back in, and it hit. Hold the last good
            # deviation instead, and let the speed limiter be the one to give
            # up (NuevoPlanNavegacion 4.3: detection brakes, it does not plan).
            self._blind += 1
            if self._blind > self.blind_ticks:
                return 0.0
            return float(np.clip(math.radians(self._prev) - carrot_bearing,
                                 -self.max_dev, self.max_dev))
        self._blind = 0

        mu_t, mu_h, mu_p = self.mu

        def ang(a, b):
            return abs((a - b + 180.0) % 360.0 - 180.0)

        cost = [mu_t * ang(c, target) + mu_h * ang(c, 0.0) + mu_p * ang(c, self._prev)
                for c in cands]
        best = float(cands[int(np.argmin(cost))])
        self._prev = best
        dev = math.radians(best) - carrot_bearing
        if self.suppress_carrot and blocked.any():
            # commanded bearing becomes `best` outright: the carrot term in
            # MotionController is goal_bearing + deviation, so cancelling it
            # means deviation = best - goal_bearing, uncapped by max_dev.
            return float(dev)
        return float(np.clip(dev, -self.max_dev, self.max_dev))

    def reset(self):
        self._binary = None
        self._prev = 0.0
        self._blind = 0
