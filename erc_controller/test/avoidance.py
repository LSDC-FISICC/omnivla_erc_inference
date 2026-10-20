#!/usr/bin/env python3
"""Pick a heading from a free-space profile. Deliberately the simplest thing.

The carrot already brings the rover back to the A* route once an obstacle is
passed -- it projects the rover onto the route and aims 1.5 m ahead, so a
deviation decays by itself. Nothing here has to plan a return.

So all this does is choose, among the bearings the camera can see, the one that
trades clearance against staying near the carrot:

    cost(b) = w_clear * shortfall(b) + w_carrot * |b - carrot|

`shortfall` is how much less than the desired lookahead that bearing offers, so
a direction with 3 m of room costs nothing and one with 0.4 m costs a lot.

This is a baseline, not the final selector. It reacts to the current frame only
and has no memory, which is exactly the thing that makes a rover oscillate once
an obstacle leaves the field of view -- the simulator's `chicane` scenario is
there to expose it.
"""
import math

import numpy as np

LOOKAHEAD_M = 1.8


class GapSelector:
    def __init__(self, lookahead_m=LOOKAHEAD_M, w_clear=1.0, w_carrot=0.5,
                 max_dev_deg=30.0, stop_below_m=0.6):
        self.lookahead = lookahead_m
        self.w_clear = w_clear
        self.w_carrot = w_carrot
        self.max_dev = math.radians(max_dev_deg)
        self.stop_below = stop_below_m

    def __call__(self, bearings_deg, free_m, carrot_bearing, caps=None):
        b = np.radians(np.asarray(bearings_deg, float))
        free = np.asarray(free_m, float)

        # only bearings the controller could actually be asked to hold
        ok = np.abs(b - 0.0) <= self.max_dev + 1e-9
        if not ok.any():
            return 0.0

        shortfall = np.clip(self.lookahead - free, 0.0, None) / self.lookahead
        toward = np.abs(np.arctan2(np.sin(b - carrot_bearing),
                                   np.cos(b - carrot_bearing))) / math.pi
        cost = self.w_clear * shortfall + self.w_carrot * toward
        cost = np.where(ok, cost, np.inf)

        best = int(np.argmin(cost))
        # If even the best direction is closed, steering is not the answer --
        # the speed limiter and GoalTurn are. Commanding a deviation here would
        # only turn a stop into a slow scrape along the obstacle.
        if free[best] < self.stop_below:
            return 0.0
        return float(np.clip(b[best], -self.max_dev, self.max_dev))
