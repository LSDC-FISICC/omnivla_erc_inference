"""Smooth driving straight from the free-space profile: steer while moving, slow down gradually.

For the drive-past-and-count plant mission on the team's own Mini+ (explore_search, drive
'reactive'). The mission still decides WHERE to go -- the open way ahead, or a frontier by A* --
and hands over the bearing of its carrot; this decides HOW: every tick, a heading among those the
rover fits through and a speed from how much room there is along it.

Why, after sidestep.py said continuous steering does not work: that was the competition rovers,
which executed ~0.36 of the commanded yaw rate while moving (2.8 m minimum radius). This unit
executes ~1.1 (config/rover_dynamics.yaml), so it can turn while it drives. What the carrot and
MPPI did with it on 14-oct (rosbags/mission_1014_1405_carrot, _1422_mppi) was stop and go: the
rover moved 23-30 % of the time, in 3 s stretches, 6-7 stops a minute -- the carrot drove at
whatever the route said and the envelope's reflexes stopped it; MPPI asked 0.07 m/s, the follower
raised that to 0.25 at the same curvature and the turn saturated at 0.3 rad/s.

Each tick:
  heading  candidates within +-cand_deg of where the profile was taken. Each has the distance the
           rover can go along it before something the profile reports comes within half_width_m
           of the line (the rover's half width plus a margin). Cost: little room, far from the
           mission's bearing, far from the last choice. The cheapest wins.
  delay    the profile is ~0.5 s old and a command acts ~1.3 s later: the heading error is taken
           from where the rover WILL be pointing -- the turn since the frame plus what the
           commands in flight will still do -- not from where it pointed when the frame was taken.
  turn     w = k_heading x that error, rate-limited, within w_max. Beyond pivot_deg (the mission
           wants to go back, say), it turns in place instead.
  speed    v_max with d_full_m or more of room along the chosen heading, down to v_min as the room
           shrinks to d_stop_m and as the heading error grows; stopped below d_stop_m, and then it
           turns towards the roomier side.

No ROS. explore_search uses it; test/test_drive_by.py and the ROS e2e (run_e2e_plants.sh) check it.
"""
import math

import numpy as np

DEFAULTS = dict(
    v_max=0.25,
    v_min=0.10,             # this unit moves from 0.10 m/s (rover_dynamics.yaml v_floor 0)
    w_max=0.35,             # rad/s commanded
    half_width_m=0.28,      # 0.125 half the rover + 0.15
    cand_deg=45.0,          # the profile is trusted within +-40 deg; a little beyond for the edge
    cand_step_deg=2.5,
    range_cap_m=3.0,
    d_stop_m=0.7,           # no advance with less room than this along the chosen heading (the
                            # envelope's own stop is 0.6: this one should act first)
    d_full_m=1.8,           # full speed with this much room
    w_clear=1.5,            # cost weights: room, the mission's bearing, the last choice
    w_goal=1.0,
    w_prev=0.4,
    k_heading=0.7,          # rad/s of turn per rad of heading error
    slow_turn_deg=45.0,     # the speed falls to v_min as the error approaches this
    pivot_deg=70.0,         # a bearing further off than this: turn in place
    pivot_w=0.30,
    w_rate=0.8,             # rad/s^2: the commanded turn changes no faster
    delay_s=1.3,            # command -> motion, as one lag (dead time 0.65 + lags)
    k_w_moving=1.1,         # what the rover does with a commanded turn while moving ...
    rest_deadband=0.10,     # ... and in place: nothing below 0.10, then 1.7 x the excess
    rest_slope=1.7,
)


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class ReactiveDrive:
    def __init__(self, **kw):
        self.p = dict(DEFAULTS)
        self.p.update({k: v for k, v in kw.items() if k in DEFAULTS})
        self._sent = []                  # (t, v, w) as returned
        self._prev_world = None          # world heading (deg) of the last choice
        self._last = (0.0, 0.0, None)    # v, w, t
        self.info = {}

    # -- geometry -------------------------------------------------------------
    def room(self, bearings_deg, ranges, hit, theta_deg):
        """How far the rover can go along theta (profile frame, deg) before a reported obstacle comes
        within half_width_m of its line; range_cap_m if nothing does."""
        p = self.p
        d = np.radians(np.asarray(bearings_deg, float) - theta_deg)
        r = np.asarray(ranges, float)
        along, across = r * np.cos(d), np.abs(r * np.sin(d))
        blk = np.asarray(hit, bool) & (along > 0.0) & (across < p['half_width_m'])
        return float(min(along[blk].min(), p['range_cap_m'])) if blk.any() else p['range_cap_m']

    def _turn_in_flight(self, t):
        """Degrees the rover will still turn on the commands of the last delay_s."""
        p = self.p
        keep = [c for c in self._sent if t - c[0] <= p['delay_s'] + 1.0]
        self._sent = keep
        seg = [c for c in keep if c[0] > t - p['delay_s']]
        before = [c for c in keep if c[0] <= t - p['delay_s']]
        if before:
            seg.insert(0, (t - p['delay_s'], before[-1][1], before[-1][2]))
        seg.append((t, 0.0, 0.0))
        turn = 0.0
        for (ta, v, w), (tb, _v, _w) in zip(seg[:-1], seg[1:]):
            if v > 0.05:
                rate = p['k_w_moving'] * w
            else:
                rate = math.copysign(p['rest_slope'] * max(0.0, abs(w) - p['rest_deadband']), w)
            turn += rate * max(0.0, tb - ta)
        return math.degrees(turn)

    # -- the step -------------------------------------------------------------
    def step(self, t, yaw_now, profile, profile_yaw, goal_deg):
        """-> (v, w). yaw_now, profile_yaw: rad, world; profile = (bearings_deg, ranges, hit) taken
        at heading profile_yaw; goal_deg: the bearing the mission wants, from the CURRENT heading."""
        p = self.p
        b, r, hit = profile
        since = math.degrees(_wrap_rad(yaw_now - profile_yaw))         # turned since the frame
        ahead = since + self._turn_in_flight(t)                        # ... and will have turned
        goal_p = goal_deg + since                                      # the goal in the profile frame
        cands = np.arange(-p['cand_deg'], p['cand_deg'] + 1e-9, p['cand_step_deg'])
        rooms = np.array([self.room(b, r, hit, c) for c in cands])
        prev_p = (_wrap_deg(self._prev_world - math.degrees(profile_yaw))
                  if self._prev_world is not None else ahead)
        cost = (p['w_clear'] * (1.0 - rooms / p['range_cap_m'])
                + p['w_goal'] * np.abs(_wrap_deg(cands - goal_p)) / 90.0
                + p['w_prev'] * np.abs(cands - prev_p) / 90.0)
        ok = rooms >= p['d_stop_m']
        if ok.any():
            cost = np.where(ok, cost, np.inf)
        i = int(np.argmin(cost))
        best, room = float(cands[i]), float(rooms[i])
        err = _wrap_deg(best - ahead)                                  # from where it will point
        goal_err = _wrap_deg(goal_p - ahead)
        v_prev, w_prev, t_prev = self._last
        dt = 0.2 if t_prev is None else min(0.5, max(0.0, t - t_prev))
        mode = 'drive'
        if abs(goal_err) > p['pivot_deg']:
            v, w, mode = 0.0, math.copysign(p['pivot_w'], goal_err), 'pivot'
        elif not ok.any():
            # nowhere to go along +-cand_deg: turn towards the roomier side, in place
            left, right = rooms[cands > 0].mean(), rooms[cands < 0].mean()
            v, w, mode = 0.0, (p['pivot_w'] if left >= right else -p['pivot_w']), 'blocked'
        else:
            w = max(-p['w_max'], min(p['w_max'], p['k_heading'] * math.radians(err)))
            s_room = (room - p['d_stop_m']) / max(p['d_full_m'] - p['d_stop_m'], 1e-6)
            s_turn = 1.0 - abs(err) / p['slow_turn_deg']
            v = p['v_max'] * min(1.0, max(0.0, s_room)) * min(1.0, max(0.0, s_turn))
            v = max(p['v_min'], v)
        if mode == 'drive':
            dw = p['w_rate'] * dt
            w = max(w_prev - dw, min(w_prev + dw, w))
        self._prev_world = math.degrees(profile_yaw) + best
        self._last = (v, w, t)
        self._sent.append((t, v, w))
        self.info = dict(mode=mode, heading=best, room=room, err=err, goal=goal_p, since=since, ahead=ahead)
        return v, w


def _wrap_rad(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
