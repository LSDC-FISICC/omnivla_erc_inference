"""What to do when the local map says the route ahead is blocked: stop, look, replan, face, and
explore if there is no path. Off by default; the checkpoint controller owns it.

Why. With local_replan alone the rover replans WHILE driving: the A* route is swapped under a
moving rover whose image is 0.4-1.2 s old (camera content 0.2-0.75 s + the 3 Hz tick + the depth
model) and whose compass lags ~0.7-1.2 s, from a map that ignores every scan taken while the
heading turns faster than 12 deg/s, with a camera that sees +-40 deg of it. On 27-sept (Wuhan
plaza) every replan left the carrot 90-110 deg off: zigzag and 23 in-place turns. When no path
exists, today nothing explores: the old route stays, the carrot pulls into the blockage and the
side-step gives up (controller_sim gardens, 28-sept).

  BRAKE     v = w = 0 for brake_s.
  LOOK      turn in place to look_deg on each side of the heading at the trigger (the goal's
            side first), stopping settle_s at each one: the map only takes scans
            while the heading is steady, and the image needs time to catch up.
  REPLAN    re-check the route on the map just built (the planner's rate limits bypassed).
            No path and explore on -> pick a frontier (seen free ground next to unseen ground,
            reachable, nearest to the goal) and route through it, at most explore_max times
            per blockage.
  FACE      if the route now starts more than face_enter_deg off the heading, turn in place
            toward it. Both turns stop commanding early by the measured turn rate x lead_s
            (as the side-step does): the rover keeps turning ~2 s after the command is cut
            (22-sept field: 37 deg, p90 50).
  -> IDLE   the carrot drives again (its controller must be reset: it recorded commands
            that were never executed).

It never runs while the side-step is mid-manoeuvre: the reflex has priority.

What controller_sim says (28-sept; 14 park, urban and standard scenarios x 6 seeds, Wuhan turn
gain 1.23 / 0.93; missions completed with no hit / hits / median time):
  today's local_replan                                          75/84  6 hits   107 s
  + local.rejoin_respects_unseen, local.unseen_extend_m 2.5,
    explore (look off)                                          84/84  0 hits    99 s
  the same + look                                               83/84  0 hits   139 s
  On the older plant (0.36) 78/84 for both today's and the first combination, 0 hits.
  The hits today: curving around a bed or planter corner at ~21 deg/s, the corner enters the
  side-step's +-20 deg sector late (0.84-0.93 m) and the 1.3 s delay carries the rover in.
  look costs 30-40 s per mission and, on the older plant, its fuller map led A* into a 1.2 m
  gap between a bed and a bench where the side-step's advance clipped the bench (2 hits).
  So the recommended setting is explore on, look off. controller_sim has no image lag, which
  is look's main benefit: that part is untested.

No ROS: checkpoint_controller_node and controller_sim import it.
"""
import math

import numpy as np
from scipy import ndimage

IDLE, BRAKE, TURN, SETTLE, REPLAN, FACE = 'idle', 'brake', 'turn', 'settle', 'replan', 'face'

DEFAULTS = dict(
    look=True,               # stop and look before trusting a replan
    explore=False,           # frontier exploration when there is no path
    brake_s=1.0,
    look_deg=60.0,           # each side; the map takes +-40 deg, so +-100 deg get seen
    turn_w=0.3,              # rad/s in place (GoalTurn's and the side-step's rate)
    lead_s=2.4,              # stop commanding when |error| < rate x lead_s (sidestep.py)
    turn_tol_deg=8.0,
    turn_timeout_s=15.0,     # a turn that does not converge (stalled) ends here
    settle_s=2.5,
    face_enter_deg=35.0,     # below this the carrot's polar steering is enough
    face_lookahead_m=1.5,    # face the route point this far ahead (the carrot distance)
    cooldown_s=8.0,          # after a recovery, ignore new triggers this long
    explore_max=3,           # frontiers per blockage episode
    explore_min_m=1.0,       # frontier distance window from the rover
    explore_max_m=8.0,
    explore_candidates=8,    # frontiers tried with A*, best first
    explore_spacing_m=1.5,   # a frontier this close to one already tried is skipped
    explore_skip_m=6.0,      # after the frontier, rejoin the route this far past the rover
    episode_progress_m=3.0,  # this much closer to the goal ends a blockage episode
)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def pick_frontier(cmap, x, y, goal, tried, p, extra_lethal=None):
    """-> (frontier (x, y), A* path) or (None, None). cmap: local_planner.LocalCostmap."""
    res = cmap.p['resolution_m']
    r = int(math.ceil(p['explore_max_m'] / res)) + 2
    (rc, cc) = cmap.cell(x, y)
    r0, r1 = max(int(rc) - r, 0), min(int(rc) + r, cmap.h - 1)
    c0, c1 = max(int(cc) - r, 0), min(int(cc) + r, cmap.w - 1)
    if r1 <= r0 or c1 <= c0:
        return None, None
    seen = cmap.seen[r0:r1 + 1, c0:c1 + 1]
    free = seen & (cmap.L[r0:r1 + 1, c0:c1 + 1] < 0.0)
    clear = cmap.distance()[r0:r1 + 1, c0:c1 + 1] > cmap.p['inflate_m'] + 0.15
    near_unseen = ndimage.binary_dilation(~seen, structure=np.ones((3, 3), bool))
    front = free & clear & near_unseen
    rows, cols = np.nonzero(front)
    if len(rows) == 0:
        return None, None
    fx, fy = cmap.centre(rows + r0, cols + c0)
    d = np.hypot(fx - x, fy - y)
    k = (d >= p['explore_min_m']) & (d <= p['explore_max_m'])
    for tx, ty in tried:
        k &= np.hypot(fx - tx, fy - ty) > p['explore_spacing_m']
    fx, fy, d = fx[k], fy[k], d[k]
    if len(fx) == 0:
        return None, None
    score = np.hypot(fx - goal[0], fy - goal[1]) + 0.5 * d
    order = np.argsort(score)
    picked = []
    for i in order:
        if len(picked) >= p['explore_candidates']:
            break
        if any(math.hypot(fx[i] - a, fy[i] - b) < p['explore_spacing_m'] for a, b in picked):
            continue
        picked.append((float(fx[i]), float(fy[i])))
        path = cmap.plan((x, y), picked[-1], extra_lethal)
        if path is not None:
            return picked[-1], path
    return None, None


class BlockedRecovery:
    def __init__(self, **kw):
        self.p = dict(DEFAULTS)
        self.p.update(kw)
        self.state = IDLE
        self.looks = 0
        self.faces = 0
        self.explores = 0
        self._cool_until = -math.inf
        self._yaws = []
        self._queue = []           # pending turn targets (absolute ENU yaw) of the look
        self._target = None
        self._t0 = 0.0
        self._after_settle = None
        self._episode = None       # (start distance to goal, tried frontiers)

    @property
    def active(self):
        return self.state != IDLE

    def _rate(self, t):
        """Measured turn rate, rad/s, over the last second of heading samples."""
        h = [(tt, a) for tt, a in self._yaws if t - tt <= 1.0]
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.3:
            return 0.0
        return abs(_wrap(h[-1][1] - h[0][1])) / (h[-1][0] - h[0][0])

    def trigger(self, t, yaw, goal_bearing, reflex_idle=True, no_path=False):
        """The planner reported a blockage (no_path: and found no way around). Starts a
        recovery unless busy, cooling down, or the side-step is mid-manoeuvre. With look off,
        only a no-path blockage starts one, and it goes straight to exploring."""
        if self.active or t < self._cool_until or not reflex_idle:
            return False
        if self.p['look']:
            side = 1.0 if goal_bearing >= 0.0 else -1.0
            a = math.radians(self.p['look_deg'])
            self._queue = [yaw + side * a, yaw - side * a]   # FACE then turns toward the new route
            self.looks += 1
        elif self.p['explore'] and no_path:
            self._queue = []
        else:
            return False
        self.state, self._t0 = BRAKE, t
        return True

    def _face_target(self, x, y, pts, cum, s_proj):
        s = min(float(cum[-1]), s_proj + self.p['face_lookahead_m'])
        px = float(np.interp(s, cum, pts[:, 0]))
        py = float(np.interp(s, cum, pts[:, 1]))
        if math.hypot(px - x, py - y) < 0.3:
            return None
        return math.atan2(py - y, px - x)

    def step(self, t, x, y, yaw, planner, pts, cum, s_proj, goal, extra_lethal=None):
        """Advance the state machine. -> (command (v, w) or None, new route pts or None, note).
        command None: the carrot drives. goal: (x, y) the leg is heading for (frontier scoring).
        extra_lethal: the static OSM layer, as for planner.check."""
        self._yaws = [(tt, a) for tt, a in self._yaws if t - tt <= 2.0] + [(t, yaw)]
        p = self.p
        note, new_pts = '', None
        if self.state == IDLE:
            return None, None, ''
        if self.state == BRAKE:
            if t - self._t0 < p['brake_s']:
                return (0.0, 0.0), None, ''
            if self._queue:
                self._next_turn(t)
                return (0.0, 0.0), None, ''
            self.state = REPLAN
        if self.state == TURN:
            err = _wrap(self._target - yaw)
            lead = self._rate(t) * p['lead_s']
            if abs(err) <= max(math.radians(p['turn_tol_deg']), lead) or t - self._t0 > p['turn_timeout_s']:
                self.state, self._t0 = SETTLE, t
                return (0.0, 0.0), None, ''
            return (0.0, math.copysign(p['turn_w'], err)), None, ''
        if self.state == SETTLE:
            if t - self._t0 < p['settle_s']:
                return (0.0, 0.0), None, ''
            if self._queue:
                self._next_turn(t)
                return (0.0, 0.0), None, ''
            if self._after_settle == FACE:
                return self._finish(t, '[recover] facing done')
            self.state = REPLAN
        if self.state == REPLAN:
            planner._last_check = -math.inf
            planner._last_plan = -math.inf
            new_pts, pnote = planner.check(t, x, y, pts, cum, s_proj, extra_lethal)
            if new_pts is not None:
                pts = new_pts
                cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
                s_proj = 0.0
                note = '[recover] replanned after looking: ' + pnote
            elif 'no path' in pnote:
                pts2, enote = self._explore(t, x, y, pts, cum, s_proj, planner, goal, extra_lethal)
                note = enote
                if pts2 is not None:
                    new_pts = pts = pts2
                    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
                    s_proj = 0.0
                else:
                    return self._finish(t, note)
            else:
                note = '[recover] route clear after looking'
            face = self._face_target(x, y, pts, cum, s_proj)
            if face is not None and abs(_wrap(face - yaw)) > math.radians(p['face_enter_deg']):
                self._queue = [face]
                self._after_settle = FACE
                self.faces += 1
                self._next_turn(t)
                return (0.0, 0.0), new_pts, note
            out = self._finish(t, note)
            return out[0], new_pts, out[2]
        return None, None, ''

    def _next_turn(self, t):
        self._target = self._queue.pop(0)
        self.state, self._t0 = TURN, t

    def _finish(self, t, note):
        self.state = IDLE
        self._after_settle = None
        self._queue = []
        self._cool_until = t + self.p['cooldown_s']
        return None, None, note

    def _explore(self, t, x, y, pts, cum, s_proj, planner, goal, extra_lethal=None):
        p = self.p
        if not p['explore']:
            return None, '[recover] no path after looking; exploration off'
        dgoal = math.hypot(goal[0] - x, goal[1] - y)
        if self._episode is None or self._episode[0] - dgoal > p['episode_progress_m']:
            self._episode = [dgoal, []]
        tried = self._episode[1]
        if len(tried) >= p['explore_max']:
            return None, f'[recover] no path; exploration budget spent ({len(tried)} frontiers)'
        f, path = pick_frontier(planner.map, x, y, goal, tried, p, extra_lethal)
        if f is None:
            return None, '[recover] no path and no reachable frontier'
        tried.append(f)
        self.explores += 1
        rest = pts[cum > s_proj + p['explore_skip_m']]
        new = np.vstack([np.asarray(path, float), rest]) if len(rest) else np.asarray(path, float)
        keep = np.concatenate([[True], np.hypot(*np.diff(new, axis=0).T) > 1e-3])
        return new[keep], (f'[recover] no path; exploring frontier {len(tried)}/{p["explore_max"]} at '
                           f'({f[0]:.1f}, {f[1]:.1f}), {math.hypot(f[0] - x, f[1] - y):.1f} m away')
