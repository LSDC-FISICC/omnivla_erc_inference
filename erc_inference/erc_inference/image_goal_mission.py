"""Image-goal checkpoint missions: reach, in order, the cone each goal image shows.

The ERC 2026 indoor track gives the mission as images, not coordinates (NYU
Indoor PDF: orange Start/Finish, red CP1, blue CP2, green CP3, yellow CP4, cone
positions changing on the day, no map). So a leg's goal is "the cone that looks
like this image", and where it stands is only known once the camera sees it.
No ROS: image_checkpoint_controller_node and test/image_goal_sim.py run this
same code.

Per leg (one goal image, reduced to its cone colour class by
erc_perception.cones.classify_goal_image):

  SCAN      turn in place in scan_step_deg steps, settling at each, watching for
            the cone. Full circle at the start of a leg, scan_partial_deg each
            side between exploration hops.
  EXPLORE   the cone was not seen: pick the most open direction in the local map
            that does not lead back to where the rover has been, drive a
            straight hop of up to explore_hop_m along it, then SCAN again.
  APPROACH  the cone is confirmed (min_confirm detections that agree): straight
            route to a standoff point standoff_m short of it, rebuilt when the
            estimate moves more than reroute_moved_m.
  ARRIVED   the estimate within arrive_m. The caller confirms with the SDK.

Driving in EXPLORE and APPROACH is the route + carrot of the outdoor stack:
local_planner.LocalReplanner splices detours around what /erc/free_space maps,
and local_recovery.BlockedRecovery (look, then replan, then face the new route,
and frontier exploration when there is no path) takes over when the route is
blocked. One map for the whole mission, in the odometry frame.

Known cones: the Start/Finish cone (next to the rover at the start) and every
cone already reached. A detection within known_cone_m of a known cone does not
count as the target, and known cones are marked in the map as obstacles. That
is also what separates red from orange, which the detector cannot tell apart:
CP1 is never the cone at the start, and the Finish leg's prior is the start.
The target cone is masked out of the free-space profile (it is 23 cm tall and
would be mapped as an obstacle), which is also why the route ends short of it.

update() -> Step(carrot, command, arrived, state, note). command (v, w)
overrides the carrot (scan turns, holds, recovery); None lets it drive.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from erc_inference import local_planner as lp
from erc_inference import local_recovery as lr

SCAN, EXPLORE, APPROACH, ARRIVED = 'scan', 'explore', 'approach', 'arrived'

DEFAULTS = dict(
    # cone fusion
    fuse_n=7,
    min_confirm=2,
    gate_m=1.5,                 # a detection within this of the estimate joins it
    known_cone_m=1.5,           # ...and within this of a known cone does not count
    start_cone_m=2.0,           # the Start/Finish cone is within this of the start pose
    max_detection_m=10.0,
    # approach
    standoff_m=1.0,
    arrive_m=1.5,
    reroute_moved_m=0.5,
    carrot_distance_m=1.5,
    cone_mask_deg=8.0,
    cone_mask_before_m=0.5,
    # scan
    scan_step_deg=90.0,         # turn this much, then settle (camera ~+-55 deg: 90 overlaps)
    scan_settle_s=1.5,          # image lag 0.4-1.2 s: let the camera catch up at each stop
    scan_partial_deg=90.0,      # each side, between exploration hops
    scan_w=0.3,
    scan_lead_s=2.4,            # the rover keeps turning ~2 s after the command stops (sidestep.py)
    scan_tol_deg=10.0,
    scan_timeout_s=12.0,        # per step; a turn that does not converge is abandoned
    # explore
    explore_hop_m=4.0,          # a hop is re-picked before it ends: corridor following, not stop-and-go
    explore_repick_s=3.0,
    scan_every_m=10.0,          # partial scan after this much exploring without one
    junction_min_m=2.0,         # a side direction (60-120 deg off) free this far is an opening:
    junction_angle_deg=60.0,    #   scan there, the cone may be down the side corridor
    explore_min_m=1.0,          # a direction must be free this far to be chosen
    explore_ray_m=8.0,
    explore_step_deg=15.0,
    hint_weight=2.0,            # m of free ray for pointing at the goal's hint (the start, for the Finish)
    visited_radius_m=2.0,       # ground within this of where the rover has been counts as visited
    visited_weight=3.0,         # m of free ray a fully visited direction costs
    turn_weight=0.5,            # m of free ray per rad of turn away from the current heading
    unseen_bonus_m=1.5,         # a ray that ends in unseen ground (a frontier) is worth this much
    hop_done_m=1.0,
    explore_goal_m=30.0,        # frontiers are scored against a point this far along the explore direction
    explore_dir_m=4.0,          # the explore direction follows the rover's motion over this much travel
    frontier_min_m=1.0,
    frontier_max_m=8.0,
    frontier_candidates=8,
    # map
    map_half_m=60.0,            # one map for the whole mission, this far around the start
)


@dataclass
class Goal:
    name: str
    cone_class: str             # erc_perception.cones class: red_orange / yellow / green / blue
    sequence: int = 0
    is_start_cone: bool = False  # the Finish: the same cone as the start


@dataclass
class Step:
    carrot: Optional[Tuple[float, float, float]]
    command: Optional[Tuple[float, float]]
    arrived: bool
    state: str
    note: str = ''
    route: Optional[np.ndarray] = field(default=None, repr=False)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _cumlen(pts):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])


def project_forward(pts, cum, x, y, s_min, window_m=10.0):
    best_s, best_d = s_min, math.inf
    for i in range(len(pts) - 1):
        if cum[i + 1] < s_min:
            continue
        if cum[i] > s_min + window_m:
            break
        a, ab = pts[i], pts[i + 1] - pts[i]
        seg2 = float(ab @ ab)
        t = 0.0 if seg2 == 0.0 else min(1.0, max(0.0, ((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / seg2))
        d = math.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1]))
        if d < best_d:
            best_d, best_s = d, cum[i] + t * math.sqrt(seg2)
    return max(s_min, best_s)


def point_at(pts, cum, s):
    """(x, y, bearing in compass degrees: 0 = +y, clockwise) at arc length s."""
    s = min(max(s, 0.0), float(cum[-1]))
    i = int(min(max(np.searchsorted(cum, s, side='right') - 1, 0), len(pts) - 2))
    a, b = pts[i], pts[i + 1]
    seg = cum[i + 1] - cum[i]
    t = 0.0 if seg <= 0.0 else (s - cum[i]) / seg
    return (float(a[0] + t * (b[0] - a[0])), float(a[1] + t * (b[1] - a[1])),
            math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0)


class ImageGoalMission:
    def __init__(self, goals: Sequence[Goal], start_xy=(0.0, 0.0), lp_params=None, recovery_params=None,
                 **kw):
        self.p = dict(DEFAULTS)
        self.p.update(kw)
        self.goals = list(goals)
        self.start = np.asarray(start_xy, float)
        h = self.p['map_half_m']
        box = np.array([[self.start[0] - h, self.start[1] - h], [self.start[0] + h, self.start[1] + h]])
        lp_kw = dict(lp_params or {})
        lp_kw['margin_m'] = 1.0
        self.replanner = lp.LocalReplanner(box, **lp_kw)
        rec = dict(look=True, explore=True)
        rec.update(recovery_params or {})
        self.recovery = lr.BlockedRecovery(**rec)
        self.known = [('start', None, self.start.copy())]      # (name, class, xy)
        self.visited: List[np.ndarray] = []
        self.index = 0
        self._arrive0 = self.p['arrive_m']
        self.notes: List[str] = []
        self._begin_leg(None)

    # -- leg bookkeeping ----------------------------------------------------

    @property
    def goal(self) -> Optional[Goal]:
        return self.goals[self.index] if self.index < len(self.goals) else None

    @property
    def done(self):
        return self.index >= len(self.goals)

    def _begin_leg(self, t):
        self.state = SCAN
        self.points: List[np.ndarray] = []
        self.pts = None
        self.cum = None
        self.s_proj = 0.0
        self._route_to = None
        self._scan_queue: List[float] = []
        self._scan_target = None
        self._scan_t0 = None
        self._scan_phase = None
        self._full_scan_pending = True
        self._arrived_note = ''
        self._hop_t0 = -math.inf
        self._last_scan_xy = None
        self._repick = False
        self._hop_heading = 0.0
        self._dir = None

    def confirm(self, t, accepted: bool):
        """The SDK's answer after ARRIVED. Accepted: next goal. Rejected: close in
        (the arrival radius halves, down to the standoff)."""
        if accepted:
            est = self.estimate()
            g = self.goal
            if est is not None and g is not None and not g.is_start_cone:
                self.known.append((g.name, g.cone_class, est.copy()))
                self._mark_cone(est)
            self.notes.append(f'{t:.1f} {g.name if g else "?"} confirmed')
            self.index += 1
            self.p['arrive_m'] = self._arrive0
            self._begin_leg(t)
        else:
            self.p['arrive_m'] = max(self.p['standoff_m'] + 0.1, 0.5 * self.p['arrive_m'])
            self.state = APPROACH
            self.notes.append(f'{t:.1f} rejected; closing in to {self.p["arrive_m"]:.1f} m')

    def _mark_cone(self, xy):
        m = self.replanner.map
        rows, cols = m.cell([xy[0]], [xy[1]])
        if m._inside(rows, cols).all():
            m.L[rows, cols] = m.p['max_logodds']
            m.seen[rows, cols] = True
            m._dist = None

    # -- inputs -------------------------------------------------------------

    def estimate(self) -> Optional[np.ndarray]:
        if len(self.points) < self.p['min_confirm']:
            return None
        return np.median(np.array(self.points[-self.p['fuse_n']:]), axis=0)

    def observe_cone(self, x, y, yaw, cone_class, bearing_deg, range_m) -> bool:
        """One detection (erc_perception.cones) seen from the estimated pose. -> counted."""
        g = self.goal
        if g is None or cone_class != g.cone_class or range_m > self.p['max_detection_m']:
            return False
        a = yaw + math.radians(bearing_deg)
        pt = np.array([x + range_m * math.cos(a), y + range_m * math.sin(a)])
        for name, cls, xy in self.known:
            if g.is_start_cone and name == 'start':
                continue
            r = self.p['start_cone_m'] if name == 'start' else self.p['known_cone_m']
            if (cls is None or cls == cone_class) and np.hypot(*(pt - xy)) <= r:
                return False
        if self.points:
            ref = np.median(np.array(self.points[-self.p['fuse_n']:]), axis=0)
            if np.hypot(*(pt - ref)) > self.p['gate_m']:
                # a second candidate: keep the nearer one (the first may have been a stray)
                if np.hypot(*(pt - (x, y))) < np.hypot(*(ref - (x, y))) - 2.0:
                    self.points = []
                else:
                    return False
        if g.is_start_cone and np.hypot(*(pt - self.start)) > 4.0:
            return False
        self.points.append(pt)
        return True

    def observe_scan(self, t, x, y, yaw, bearings_deg, ranges, hit) -> bool:
        b = np.asarray(bearings_deg, float)
        r = np.asarray(ranges, float)
        h = np.asarray(hit, bool).copy()
        est = self.estimate()
        if est is not None:
            d = float(np.hypot(*(est - (x, y))))
            brg = math.degrees(_wrap(math.atan2(est[1] - y, est[0] - x) - yaw))
            half = max(self.p['cone_mask_deg'], math.degrees(math.atan2(0.3, max(d, 0.3))))
            h[(np.abs(b - brg) <= half) & (r >= d - self.p['cone_mask_before_m'])] = False
        return self.replanner.observe(t, x, y, yaw, b, r, h)

    # -- scan ---------------------------------------------------------------

    def _plan_scan(self, yaw, full):
        step = math.radians(self.p['scan_step_deg'])
        if full:
            n = int(round(2 * math.pi / step))
            self._scan_queue = [yaw + step * (i + 1) for i in range(n)]
        else:
            a = math.radians(self.p['scan_partial_deg'])
            self._scan_queue = [yaw + a, yaw - a, yaw]
        self._scan_target = None
        self._scan_phase = None

    def _scan_step(self, t, yaw, rate):
        p = self.p
        if self._scan_target is None:
            if not self._scan_queue:
                return None
            self._scan_target = self._scan_queue.pop(0)
            self._scan_phase, self._scan_t0 = 'turn', t
        if self._scan_phase == 'turn':
            err = _wrap(self._scan_target - yaw)
            if abs(err) <= max(math.radians(p['scan_tol_deg']), rate * p['scan_lead_s']) \
                    or t - self._scan_t0 > p['scan_timeout_s']:
                self._scan_phase, self._scan_t0 = 'settle', t
                return (0.0, 0.0)
            return (0.0, math.copysign(p['scan_w'], err))
        if t - self._scan_t0 < p['scan_settle_s']:
            return (0.0, 0.0)
        self._scan_target = None
        return self._scan_step(t, yaw, rate)

    # -- explore ------------------------------------------------------------

    def _explore_heading(self, x, y, yaw, hint=None):
        p = self.p
        m = self.replanner.map
        best, best_score = None, -math.inf
        side_open = False
        vis = np.array(self.visited[-400:]) if self.visited else np.zeros((0, 2))
        steps = np.arange(0.2, p['explore_ray_m'] + 1e-9, 0.2)
        for a in np.arange(-math.pi, math.pi, math.radians(p['explore_step_deg'])):
            ang = yaw + a
            xs, ys = x + np.cos(ang) * steps, y + np.sin(ang) * steps
            clear = m.clearance_at(xs, ys) > m.p['inflate_m']
            rows, cols = m.cell(xs, ys)
            ok = m._inside(rows, cols)
            seen = np.zeros(len(xs), bool)
            seen[ok] = m.seen[rows[ok], cols[ok]]
            blocked = ~clear | ~ok
            n_free = int(np.argmax(blocked)) if blocked.any() else len(steps)
            free_len = float(steps[n_free - 1]) if n_free > 0 else 0.0
            if free_len < p['explore_min_m']:
                continue
            if math.radians(p['junction_angle_deg']) <= abs(a) <= math.radians(180 - p['junction_angle_deg']) \
                    and free_len >= p['junction_min_m']:
                end = np.array([x + math.cos(ang) * free_len, y + math.sin(ang) * free_len])
                if not len(vis) or np.hypot(*(vis - end).T).min() > p['visited_radius_m']:
                    side_open = True
            ends_unseen = n_free < len(steps) and not seen[min(n_free, len(steps) - 1)] or \
                (n_free > 0 and not seen[n_free - 1])
            score = free_len + (p['unseen_bonus_m'] if ends_unseen else 0.0) - p['turn_weight'] * abs(a)
            if hint is not None and np.hypot(*(hint - (x, y))) > 1.0:
                score += p['hint_weight'] * math.cos(_wrap(ang - math.atan2(hint[1] - y, hint[0] - x)))
            if len(vis):
                end = np.array([x + math.cos(ang) * min(free_len, p['explore_hop_m']),
                                y + math.sin(ang) * min(free_len, p['explore_hop_m'])])
                near = np.hypot(*(vis - end).T) < p['visited_radius_m']
                score -= p['visited_weight'] * float(near.mean() > 0) * (1.0 + float(near.mean()))
            if score > best_score:
                best, best_score, best_len = ang, score, free_len
        if best is None:
            return None, 0.0, False
        return best, min(best_len, p['explore_hop_m']), side_open

    def _set_route(self, pts):
        self.pts = np.asarray(pts, float)
        self.cum = _cumlen(self.pts)
        self.s_proj = 0.0

    # -- the step -----------------------------------------------------------

    def update(self, t, x, y, yaw, rate=0.0, reflex_idle=True) -> Step:
        """rate: measured turn rate, rad/s (for the scan's lead). reflex_idle: False while
        a side-step is manoeuvring (recovery waits for it)."""
        p = self.p
        if self.done:
            return Step(None, (0.0, 0.0), False, 'done', 'mission complete')
        if not self.visited or np.hypot(*(self.visited[-1] - (x, y))) > 0.5:
            self.visited.append(np.array([x, y]))
        est = self.estimate()
        g = self.goal
        if self.state == APPROACH and est is None:
            self.state = SCAN          # the estimate was reset (a nearer candidate): look again
        if self.state == ARRIVED:
            return Step(None, (0.0, 0.0), True, ARRIVED, self._arrived_note)
        if est is not None and self.state != APPROACH and not self.recovery.active:
            self.state = APPROACH
            self._scan_queue, self._scan_target = [], None
            self.notes.append(f'{t:.1f} {g.name}: cone at ({est[0]:.1f},{est[1]:.1f})')
        if est is not None and math.hypot(est[0] - x, est[1] - y) <= p['arrive_m']:
            self.state = ARRIVED
            self._arrived_note = f'{g.name}: {math.hypot(est[0] - x, est[1] - y):.1f} m from the cone'
            self.notes.append(f'{t:.1f} arrived {self._arrived_note}')
            return Step(None, (0.0, 0.0), True, ARRIVED, self._arrived_note)

        if self.state == SCAN:
            if self._scan_target is None and not self._scan_queue:
                self._plan_scan(yaw, self._full_scan_pending)
                self._full_scan_pending = False
            cmd = self._scan_step(t, yaw, rate)
            if cmd is not None:
                return Step(None, cmd, False, SCAN, 'scanning')
            self._last_scan_xy = np.array([x, y])
            if not self._pick_hop(t, x, y, yaw):
                self._full_scan_pending = True
                self.notes.append(f'{t:.1f} explore: no open direction; scanning again')
                return Step(None, (0.0, 0.0), False, SCAN, 'no open direction')
            self.state = EXPLORE

        if self.state == APPROACH:
            if self._route_to is None or np.hypot(*(est - self._route_to)) > p['reroute_moved_m']:
                v = est - np.array([x, y])
                d = float(np.hypot(*v))
                end = est - v / d * p['standoff_m'] if d > p['standoff_m'] + 0.05 else np.array([x, y]) + 0.05 * v / max(d, 1e-6)
                self._set_route([(x, y), end])
                self._route_to = est.copy()

        # EXPLORE and APPROACH drive the route; the planner and the recovery keep it clear
        self.s_proj = project_forward(self.pts, self.cum, x, y, self.s_proj)
        goal_xy = est if est is not None else self.pts[-1]
        new, note = (None, '')
        if not self.recovery.active:
            new, note = self.replanner.check(t, x, y, self.pts, self.cum, self.s_proj)
            if new is not None:
                self._set_route(new)
            if note.startswith('[local-plan] blocked'):
                gb = _wrap(math.atan2(goal_xy[1] - y, goal_xy[0] - x) - yaw)
                self.recovery.trigger(t, yaw, gb, reflex_idle=reflex_idle, no_path='no path' in note)
            if note:
                self.notes.append(f'{t:.1f} {note}')
        cmd, new2, rnote = self.recovery.step(t, x, y, yaw, self.replanner, self.pts, self.cum, self.s_proj,
                                              tuple(goal_xy))
        if new2 is not None:
            self._set_route(new2)
        if rnote:
            self.notes.append(f'{t:.1f} {rnote}')
        if self.recovery.active:
            return Step(self._carrot(), cmd if cmd is not None else (0.0, 0.0), False, self.state, 'recovering')
        if self.state == EXPLORE:
            if rnote.startswith('[recover]') and ('budget spent' in rnote or 'no reachable' in rnote
                                                  or 'exploration off' in rnote):
                self._repick = True           # the recovery gave up on this hop: a new direction
            near_end = float(self.cum[-1]) - self.s_proj < p['hop_done_m']
            if near_end or self._repick or t - self._hop_t0 >= p['explore_repick_s']:
                since = np.hypot(*(np.array([x, y]) - self._last_scan_xy)) if self._last_scan_xy is not None else 1e9
                heading, hop, side_open = self._explore_heading(x, y, yaw, self.start if g.is_start_cone else None)
                if since >= p['scan_every_m'] or heading is None:
                    self.state = SCAN
                    why = 'side opening' if side_open else ('no open direction' if heading is None else 'distance')
                    self.notes.append(f'{t:.1f} partial scan ({why})')
                    return Step(None, (0.0, 0.0), False, SCAN, 'scan')
                self._pick_hop(t, x, y, yaw)
        return Step(self._carrot(), None, False, self.state, note, self.pts)

    def _explore_dir(self, yaw):
        """Unit direction to explore: the way the rover has been moving (its trail over the last
        explore_dir_m), else the committed one, else its heading."""
        trail = self.visited[-int(self.p['explore_dir_m'] / 0.5) - 1:]
        if len(trail) >= 3:
            v = trail[-1] - trail[0]
            if np.hypot(*v) > 0.5 * self.p['explore_dir_m']:
                return v / np.hypot(*v)
        if self._dir is not None:
            return self._dir
        return np.array([math.cos(yaw), math.sin(yaw)])

    def _pick_hop(self, t, x, y, yaw, choice=None):
        """Route to the reachable frontier nearest a point far along the explore direction
        (local_recovery.pick_frontier: seen free ground next to unseen, A* path). None ahead:
        try the two sides, then back. The start cone's hint (the Finish) pulls the point."""
        p = self.p
        g = self.goal
        d0 = self._explore_dir(yaw)
        fp = dict(explore_min_m=p['frontier_min_m'], explore_max_m=p['frontier_max_m'],
                  explore_candidates=p['frontier_candidates'], explore_spacing_m=1.5)
        for k, d in enumerate([d0, np.array([-d0[1], d0[0]]), np.array([d0[1], -d0[0]]), -d0]):
            goal = np.array([x, y]) + p['explore_goal_m'] * d
            if g.is_start_cone:
                goal = 0.5 * goal + 0.5 * self.start
            f, path = lr.pick_frontier(self.replanner.map, x, y, goal, [], fp)
            if f is not None:
                self._set_route(path if len(path) >= 2 else [(x, y), f])
                self._dir = d
                self._hop_heading = math.atan2(f[1] - y, f[0] - x)
                self._hop_t0, self._repick = t, False
                self.notes.append(f'{t:.1f} explore: frontier ({f[0]:.1f},{f[1]:.1f}) '
                                  f'{["ahead", "left", "right", "back"][k]}')
                return True
        # nothing reachable: fall back to the most open direction
        heading, hop = choice if choice is not None else \
            self._explore_heading(x, y, yaw, self.start if g.is_start_cone else None)[:2]
        if heading is None:
            return False
        self._set_route([(x, y), (x + hop * math.cos(heading), y + hop * math.sin(heading))])
        self._hop_heading, self._hop_t0, self._repick = heading, t, False
        self.notes.append(f'{t:.1f} explore {math.degrees(_wrap(heading)):.0f} deg for {hop:.1f} m (no frontier)')
        return True

    def _carrot(self):
        return point_at(self.pts, self.cum, self.s_proj + self.p['carrot_distance_m'])
