"""Search mission with NO known map: build the 2D map from the free-space profile, explore it,
put every object found on it, and visit each one.

For "drive around indoors and find the potted plants" (object_detector_node, YOLO26). Where
loop_patrol.LoopPatrolMission patrols corridors it was given, this one has only what the rover
sees: the same log-odds grid local_planner.LocalCostmap builds from /erc/free_space, kept for
the whole mission in the mission frame (origin at the start pose), and the objects' positions
as markers on it (loop_patrol.ConeMap: detections clustered by class). No ROS:
image_checkpoint_controller_node (param explore) and test/explore_search_sim.py run this code,
through the same interface as LoopPatrolMission.

  EXPLORE  go to a frontier: free, seen ground next to ground the camera has not looked at,
           the rover can get to, chosen ahead of it when there is one near, by A* on the map. Reached, or no longer a frontier -> the next. None left near the
           rover -> look around once; none anywhere -> the area is explored.
  GOTO     an object is on the map (min_confirm detections): A* to a standoff point short of
           it, the nearest unvisited first; the route is rebuilt as the estimate moves.
  SHOW     within arrive_m: face it, hold still show_s. ARRIVED -> the caller calls confirm().
  SCAN     turns in place: at the start, and when there is no frontier in reach.
  DONE     as many objects shown as there are goals (or patrol.max_shows), or nothing left to
           explore (stop_when_explored).

What it does not have, and what follows from that:
  * No drift correction. The map and the pose drift together, so what is near the rover is
    consistent with what it sees, but a place mapped minutes ago is displaced when the rover
    comes back to it (walls drawn twice, an object mapped twice: same_class_apart_m covers
    the second). Nothing here closes loops.
  * The profile reaches 3 m in +-60 deg and only +-map_fov_deg enters the map: rooms wider
    than ~5 m are explored along their walls, slowly.
  * Phantom obstacles of the depth model become walls here; hit evidence decays only when a
    later ray crosses the cell as free.
"""
import math
from typing import List, Optional, Sequence

import numpy as np
from scipy import ndimage

from erc_inference import local_planner as lp
from erc_inference.image_goal_mission import Goal, _cumlen, _wrap, point_at, project_forward
from erc_inference.loop_patrol import ARRIVED, DONE, GOTO, SCAN, SHOW, ConeCluster, ConeMap, Step
from erc_inference.loop_patrol import DEFAULTS as PATROL_DEFAULTS

EXPLORE = 'explore'

DEFAULTS = dict(
    map_half_m=30.0,            # the map is a square this far each way from the start pose
    frontier_min_m=1.0,         # frontiers nearer than this are not worth a route
    frontier_near_m=6.0,        # looked for within this first (keeps the rover going where it is)...
    frontier_far_m=40.0,        # ...then anywhere on the map
    frontier_ahead_m=4.0,       # prefer frontiers near the point this far ahead of the rover
    frontier_reach_m=0.8,       # this close to the frontier it is reached
    frontier_timeout_s=90.0,    # a frontier not reached in this long is dropped
    frontier_candidates=8,      # frontiers tried with A*, best first, near and then far
    frontier_spacing_m=1.5,     # one tried (reached or failed) is not picked again this close...
    frontier_memory_s=180.0,    # ...for this long
    goto_angles_deg=(0.0, 35.0, -35.0, 70.0, -70.0, 110.0, -110.0, 180.0),   # standoff points tried
                                # around the object, from the rover's side
    unreachable_s=60.0,         # an object with no path to any of them is left alone this long
    stop_when_explored=True,    # no frontier anywhere and no object pending: the mission ends
    route_clear_m=4.0,          # route length ahead checked against the map
    object_gate_m=1.5,          # detections this close to the object being approached join it (it follows
                                # the live view). loop_patrol's target_gate_m is 4 m, for cones of different
                                # colours far apart; here every object is of one class, and at 4 m the plant
                                # beside the target was taken for it and dragged its marker (ROS e2e, 9-oct:
                                # one plant shown twice)
    stuck_turning_deg=40.0,     # with the carrot this far off the heading the rover is turning in place
                                # (the carrot controller does from 45 deg), and not moving is expected: a
                                # frontier behind it takes > 12 s to face at that turn rate with the command
                                # delay, and read as "stuck" with nothing ahead (ROS e2e, 9-oct). Only for
    stuck_turn_grace_s=30.0,    # this long: a turn that never ends is being stuck too
)


class ExploreSearchMission:
    def __init__(self, goals: Sequence[Goal], start_xy=(0.0, 0.0), lp_params=None, **kw):
        self.p = dict(PATROL_DEFAULTS)
        self.p.update(DEFAULTS)
        self.p.update(kw)
        self.p['any_order'] = True          # objects are visited as they are found
        self.goals = list(goals)
        self.start = np.asarray(start_xy, float)
        h = float(self.p['map_half_m'])
        lp_kw = dict(lp_params or {})
        lp_kw['margin_m'] = 0.0
        self.replanner = lp.LocalReplanner(np.array([self.start - h, self.start + h]), **lp_kw)
        self.map = self.replanner.map
        self.cones = ConeMap(self.p)        # the objects; the name is loop_patrol's
        self.index = 0
        self._arrive0 = self.p['arrive_m']
        self.notes: List[str] = []
        self.state = EXPLORE
        self.pts = self.cum = None
        self.s_proj = 0.0
        self._route_kind = None
        self._route_t = -math.inf
        self._route_to = None
        self.target: Optional[ConeCluster] = None
        self.frontier = None
        self._frontier_t = 0.0
        self._tried = []                 # (t, x, y) frontiers reached or given up
        self._skip = {}                  # id(cluster) -> not before t (no path to it)
        self._scan_queue = []
        self._scan_target = None
        self._scan_phase = None
        self._scan_t0 = 0.0
        self._scanned_at = None          # where the last no-frontier scan was made
        self._yaws = []
        self._trail = []
        self._turning_since = None
        self._last_profile = None
        self._last_check = -math.inf
        self._backoff_until = None
        self.stucks = 0
        self.explored = False
        self._show_phase, self._show_t0 = None, 0.0
        if self.p['start_scan']:
            # there and back, as loop_patrol: the turns cancel, and with them the gyro's scale error
            q = math.pi / 2
            self._scan_queue = [q, 2 * q, q, 0.0, -q, 0.0]
            self.state = SCAN

    # -- the interface image_checkpoint_controller_node uses -------------------

    @property
    def goal(self) -> Optional[Goal]:
        if self.index < len(self.goals):
            return self.goals[self.index]
        if not self.done:
            return Goal(f'extra{self.index + 1}', 'any', self.index + 1)
        return None

    @property
    def n_shows(self):
        return max(int(self.p.get('max_shows', 0) or 0), len(self.goals))

    @property
    def done(self):
        return self.index >= self.n_shows or self.explored

    def turn_rate_dps(self):
        h = self._yaws
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.3:
            return 0.0
        return abs(math.degrees(_wrap(h[-1][1] - h[0][1]))) / (h[-1][0] - h[0][0])

    def observe_cone(self, x, y, yaw, cls, bearing_deg, range_m, t=None):
        if range_m > self.p['max_detection_m']:
            return None
        if t is not None:
            self._yaws = [(tt, a) for tt, a in self._yaws if t - tt <= 1.0] + [(t, yaw)]
        if self.turn_rate_dps() > self.p['cone_max_turn_dps']:
            return None
        a = yaw + math.radians(bearing_deg)
        return self.cones.add(cls, (x + range_m * math.cos(a), y + range_m * math.sin(a)), t,
                              prefer=self.target, prefer_gate=self.p['object_gate_m'])

    def observe_scan(self, t, x, y, yaw, bearings_deg, ranges, hit):
        b, r, h = np.asarray(bearings_deg, float), np.asarray(ranges, float), np.asarray(hit, bool)
        self._last_profile = (b, r, h)
        self.replanner.observe(t, x, y, yaw, b, r, h)

    def confirm(self, t, accepted: bool):
        if self.target is not None:
            self.target.visited = True
            self.target.name = f'cone{self.index + 1}' if accepted else 'rejected'
        self.notes.append(f'{t:.1f} {self.goal.name} ' + ('confirmed' if accepted else 'rejected'))
        if accepted:
            self.index += 1
        self.target = None
        self.state = EXPLORE if not self.done else DONE
        self._route_kind = None

    def block_ahead(self, t, x, y, yaw, why):
        """Something ahead stopped the rover (stuck, or tilting onto it): back off, mark it, reroute."""
        p = self.p
        self.stucks += 1
        self._backoff_until = t + p['backoff_s']
        ahead = None
        if self._last_profile is not None:
            b, r, h = self._last_profile
            k = (np.abs(b) <= 20.0) & h
            ahead = float(r[k].min()) if k.any() else None
        mark = why.startswith('tilt') or (ahead is not None and ahead < p['stuck_mark_to_m'] + 0.1)
        if mark:
            m = self.map
            ds = np.arange(p['stuck_mark_from_m'], p['stuck_mark_to_m'] + 1e-9, m.p['resolution_m'] * 0.5)
            dl = np.arange(-p['stuck_mark_half_m'], p['stuck_mark_half_m'] + 1e-9, m.p['resolution_m'] * 0.5)
            S, Lg = np.meshgrid(ds, dl)
            rows, cols = m.cell((x + S * math.cos(yaw) - Lg * math.sin(yaw)).ravel(),
                                (y + S * math.sin(yaw) + Lg * math.cos(yaw)).ravel())
            ok = m._inside(rows, cols)
            m.L[rows[ok], cols[ok]] = m.p['max_logodds']
            m.seen[rows[ok], cols[ok]] = True
            m._dist = None
        if self.frontier is not None:
            self._tried.append((t, *self.frontier))
            self.frontier = None
        self._route_kind = None
        self._trail = []
        self.notes.append(f'{t:.1f} {why}: backing off, ' + ('obstacle marked ahead' if mark else 'nothing seen ahead')
                          + ', rerouting')

    def export_state(self):
        return {'index': self.index,
                'cones': [{'cls': c.cls, 'name': c.name, 'visited': c.visited,
                           'points': [list(map(float, pt)) for pt in c.points[-self.p['fuse_n']:]]}
                          for c in self.cones.clusters if c.points]}

    def import_state(self, state, first_goal_index=None):
        """The objects and the count survive a restart of the node; the map is built again."""
        self.index = min(int(state.get('index', 0)), self.n_shows)
        self.cones.clusters = [
            ConeCluster(c['cls'], [np.asarray(q, float) for q in c['points']], name=c.get('name', ''),
                        visited=bool(c.get('visited', False)), times=[-math.inf] * len(c['points']))
            for c in state.get('cones', [])]
        self.target = None
        self.state = EXPLORE if not self.done else DONE
        self._route_kind = None
        self.notes.append(f'resumed at {self.index + 1}/{self.n_shows} with {len(self.cones.clusters)} objects mapped')

    # -- for looking at (the node publishes these) -----------------------------

    def objects(self):
        """[(class, x, y, visited, detections)] of every confirmed object on the map."""
        return [(c.cls, float(c.estimate()[0]), float(c.estimate()[1]), bool(c.visited and c.name != 'rejected'),
                 len(c.points)) for c in self.cones.clusters if len(c.points) >= self.p['min_confirm']]

    # -- routes ----------------------------------------------------------------

    def _set_route(self, pts, kind, t, to=None):
        self.pts = np.asarray(pts, float)
        self.cum = _cumlen(self.pts)
        self.s_proj = 0.0
        self._route_kind = kind
        self._route_t = t
        self._route_to = None if to is None else np.asarray(to, float).copy()
        # not self._trail = []: the route to an object is rebuilt every reroute_every_s, and clearing
        # the stuck window with it meant a rover held by a door jamb was never "stuck" (sim, 9-oct)

    def _carrot(self):
        return point_at(self.pts, self.cum, self.s_proj + self.p['carrot_distance_m'])

    def _route_blocked(self, t):
        """True when the route ahead crosses what the map now holds as an obstacle."""
        p = self.replanner.p
        if t - self._last_check < p['check_period_s']:
            return False
        self._last_check = t
        s1 = min(float(self.cum[-1]), self.s_proj + self.p['route_clear_m'])
        s0 = min(self.s_proj + p['ignore_within_m'], s1)
        if s1 - s0 < 0.05:
            return False
        ss = np.arange(s0, s1 + 1e-9, 0.5 * p['resolution_m'])
        xy = np.array([point_at(self.pts, self.cum, s)[:2] for s in ss])
        return bool((self.map.clearance_at(xy[:, 0], xy[:, 1]) <= p['inflate_m']).any())

    def _route_to_object(self, t, x, y, est):
        d = math.atan2(y - est[1], x - est[0])
        for a in self.p['goto_angles_deg']:
            ang = d + math.radians(a)
            end = (est[0] + self.p['standoff_m'] * math.cos(ang), est[1] + self.p['standoff_m'] * math.sin(ang))
            if math.hypot(end[0] - x, end[1] - y) < 0.3:
                path = [(x, y), end]
            else:
                path = self.map.plan((x, y), end)
            if path is not None:
                self._set_route(path if len(path) >= 2 else [(x, y), end], GOTO, t, est)
                return True
        return False

    def _frontiers(self, x, y):
        """Frontier cells the rover can get to: (xs, ys) of free, seen ground next to unseen ground,
        in the same connected piece of passable ground as the rover.

        Connected, because a ray that slips past a wall (the profile misses ~3% of what is there)
        leaves free cells on its far side: frontiers no path leads to. With only the best few tried
        by A*, a handful of those hid every real one and the mission ended "explored" with a room
        never entered (explore_search_sim seed 4, 9-oct)."""
        m = self.map
        passable = m.seen & (m.L < 0.0) & (m.distance() > m.p['inflate_m'])
        labels, n = ndimage.label(passable, structure=np.ones((3, 3), bool))
        if n == 0:
            return np.zeros(0), np.zeros(0)
        r, c = (int(v) for v in m.cell(x, y))
        k = int(math.ceil(0.8 / m.p['resolution_m']))        # the rover may stand inside the inflation
        r0, c0 = max(r - k, 0), max(c - k, 0)
        near = labels[r0:r + k + 1, c0:c + k + 1]
        rr, cc = np.nonzero(near)
        if len(rr) == 0:
            return np.zeros(0), np.zeros(0)
        i = int(np.argmin((rr + r0 - r) ** 2 + (cc + c0 - c) ** 2))
        mine = labels == near[rr[i], cc[i]]
        front = mine & ndimage.binary_dilation(~m.seen, structure=np.ones((3, 3), bool)) \
            & (m.distance() > m.p['inflate_m'] + 0.15)
        rows, cols = np.nonzero(front)
        return m.centre(rows, cols)

    def _pick_frontier(self, t, x, y, yaw):
        p = self.p
        self._tried = [e for e in self._tried if t - e[0] <= p['frontier_memory_s']]
        fx, fy = self._frontiers(x, y)
        self.frontier = None
        if len(fx) == 0:
            return False
        d = np.hypot(fx - x, fy - y)
        keep = d >= p['frontier_min_m']
        for _, tx, ty in self._tried:
            keep &= np.hypot(fx - tx, fy - ty) > p['frontier_spacing_m']
        fx, fy, d = fx[keep], fy[keep], d[keep]
        ahead = (x + p['frontier_ahead_m'] * math.cos(yaw), y + p['frontier_ahead_m'] * math.sin(yaw))
        for reach in (p['frontier_near_m'], p['frontier_far_m']):
            k = d <= reach
            if not k.any():
                continue
            gx, gy, gd = fx[k], fy[k], d[k]
            # near: the one ahead (keep going where it is going); far: the nearest (it has to go back)
            score = np.hypot(gx - ahead[0], gy - ahead[1]) + 0.5 * gd if reach == p['frontier_near_m'] else gd
            picked = []
            for i in np.argsort(score):
                if len(picked) >= p['frontier_candidates']:
                    break
                f = (float(gx[i]), float(gy[i]))
                if any(math.hypot(f[0] - a, f[1] - b) < p['frontier_spacing_m'] for a, b in picked):
                    continue
                picked.append(f)
                path = self.map.plan((x, y), f)
                if path is not None and len(path) >= 2:
                    self.frontier, self._frontier_t = f, t
                    self._set_route(path, EXPLORE, t, f)
                    return True
        return False

    def _frontier_done(self, t, x, y):
        """None while the frontier is still worth going to; else 'reached' (there, or seen from here:
        nothing unseen left around it) or 'failed' (not reached in frontier_timeout_s)."""
        f = self.frontier
        if f is None or self._route_kind != EXPLORE:
            return 'reached'
        if math.hypot(f[0] - x, f[1] - y) <= self.p['frontier_reach_m'] or float(self.cum[-1]) - self.s_proj < 0.3:
            return 'reached'
        if t - self._frontier_t > self.p['frontier_timeout_s']:
            self.notes.append(f'{t:.1f} frontier ({f[0]:.1f},{f[1]:.1f}) not reached in '
                              f'{self.p["frontier_timeout_s"]:.0f} s: dropped')
            return 'failed'
        r, c = self.map.cell(f[0], f[1])
        r, c = int(r), int(c)
        return 'reached' if self.map.seen[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].all() else None

    # -- scans -------------------------------------------------------------------

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

    def _check_stuck(self, t, x, y, yaw):
        p = self.p
        c = self._carrot()
        off = abs(_wrap(math.atan2(c[1] - y, c[0] - x) - yaw))
        if off > math.radians(p['stuck_turning_deg']):
            if self._turning_since is None:
                self._turning_since = t
            if t - self._turning_since <= p['stuck_turn_grace_s']:
                self._trail = []
                return False
        else:
            self._turning_since = None
        self._trail = [e for e in self._trail if t - e[0] <= p['stuck_window_s']] + [(t, x, y)]
        if self._trail[-1][0] - self._trail[0][0] < p['stuck_window_s'] - 0.5:
            return False
        xs = np.array([e[1] for e in self._trail])
        ys = np.array([e[2] for e in self._trail])
        return float(np.hypot(xs - xs[0], ys - ys[0]).max()) < p['stuck_min_move_m']

    # -- the step ----------------------------------------------------------------

    def update(self, t, x, y, yaw, rate=0.0, reflex_idle=True) -> Step:
        p = self.p
        if self.done:
            return Step(None, (0.0, 0.0), False, DONE, 'explored' if self.explored else 'mission complete')
        self._yaws = [(tt, a) for tt, a in self._yaws if t - tt <= 1.0] + [(t, yaw)]
        if self.target is None:
            shown = [c for c in self.cones.clusters if c.visited and c.name.startswith('cone')]

            def fresh(c):
                e = c.estimate()
                return all(v.cls != c.cls or float(np.hypot(*(v.estimate() - e))) > p['same_class_apart_m']
                           for v in shown)
            cands = [c for c in self.cones.clusters if not c.visited and len(c.points) >= p['min_confirm']
                     and fresh(c) and t >= self._skip.get(id(c), -math.inf)]
            if cands:
                self.target = min(cands, key=lambda c: float(np.hypot(*(c.estimate() - (x, y)))))
                e = self.target.estimate()
                self.notes.append(f'{t:.1f} object {self.index + 1}/{self.n_shows}: {self.target.cls} at '
                                  f'({e[0]:.1f},{e[1]:.1f})')
                self._route_kind = None
        est = None
        if self.target is not None:
            est = self.target.live(t, p['live_s'])
            if est is None:
                est = self.target.estimate(p['fuse_n'])
        if self.state == ARRIVED:
            return Step(None, (0.0, 0.0), True, ARRIVED)
        if self.state == SHOW:
            return self._show(t, x, y, yaw, est, rate)
        if est is not None and math.hypot(est[0] - x, est[1] - y) <= p['arrive_m']:
            self.state = SHOW
            self._show_phase, self._show_t0 = 'face', t
            self.notes.append(f'{t:.1f} at the {self.target.cls}: {math.hypot(est[0] - x, est[1] - y):.1f} m; showing it')
            return self._show(t, x, y, yaw, est, rate)
        if self._backoff_until is not None:
            if t < self._backoff_until:
                return Step(None, (p['backoff_v'], 0.0), False, self.state, 'backing off')
            self._backoff_until = None
        if self.state == SCAN:
            if est is not None:              # something to go to: the look-around can wait
                self._scan_queue, self._scan_target = [], None
            cmd = self._scan_step(t, yaw, 0.0 if rate is None else rate)
            if cmd is not None:
                self._trail = []
                return Step(None, cmd, False, SCAN, 'scanning')
            self.state = EXPLORE
            self._route_kind = None
        if est is not None:
            self.state = GOTO
            if self._route_kind != GOTO or np.hypot(*(est - self._route_to)) > p['reroute_moved_m'] \
                    or t - self._route_t > p['reroute_every_s'] or self._route_blocked(t):
                if not self._route_to_object(t, x, y, est):
                    self._skip[id(self.target)] = t + p['unreachable_s']
                    self.notes.append(f'{t:.1f} no path to the {self.target.cls} at ({est[0]:.1f},{est[1]:.1f}); '
                                      f'exploring on')
                    self.target = None
                    self.state = EXPLORE
                    self._route_kind = None
                    return Step(None, (0.0, 0.0), False, EXPLORE, 'no path to the object')
        else:
            self.state = EXPLORE
            if self._route_kind == EXPLORE and self._route_blocked(t):
                path = self.map.plan((x, y), self.frontier)       # another way to the same frontier
                if path is not None and len(path) >= 2:
                    self._set_route(path, EXPLORE, t, self.frontier)
                else:
                    self._tried.append((t, *self.frontier))
                    self.frontier = None
                    self._route_kind = None
            how = self._frontier_done(t, x, y)
            if how is not None:
                # only one that FAILED is remembered: one reached is no frontier any more, and
                # remembering it kept its neighbours (within frontier_spacing_m) from being chosen --
                # the rover stood "looking around" for minutes with frontiers 3 m ahead (sim, 9-oct)
                if how == 'failed' and self.frontier is not None:
                    self._tried.append((t, *self.frontier))
                if not self._pick_frontier(t, x, y, yaw):
                    here = np.array([x, y])
                    if self._scanned_at is None or np.hypot(*(here - self._scanned_at)) > 1.0:
                        # nothing in reach from what this heading showed: look around once
                        self._scanned_at = here
                        q = math.pi / 2
                        self._scan_queue = [yaw + q, yaw + 2 * q, yaw + q, yaw, yaw - q, yaw]
                        self._scan_target = None
                        self.state = SCAN
                        self.notes.append(f'{t:.1f} no frontier in reach: looking around')
                        return Step(None, (0.0, 0.0), False, SCAN, 'scan')
                    if self._tried:
                        self._tried = []         # everything else was tried: try them again once
                        self._scanned_at = None
                        return Step(None, (0.0, 0.0), False, EXPLORE, 'no frontier')
                    if p['stop_when_explored']:
                        self.explored = True
                        self.notes.append(f'{t:.1f} nothing left to explore: {self.index}/{self.n_shows} objects shown')
                        return Step(None, (0.0, 0.0), False, DONE, 'explored')
                    return Step(None, (0.0, 0.0), False, EXPLORE, 'no frontier')
        self.s_proj = project_forward(self.pts, self.cum, x, y, self.s_proj)
        if self._check_stuck(t, x, y, yaw):
            self.block_ahead(t, x, y, yaw, 'stuck')
            return Step(None, (p['backoff_v'], 0.0), False, self.state, 'backing off')
        return Step(self._carrot(), None, False, self.state, '', self.pts)

    def _show(self, t, x, y, yaw, est, rate):
        """Face the object (well inside the image), then hold still show_s."""
        p = self.p
        if self._show_phase == 'face' and est is not None:
            err = _wrap(math.atan2(est[1] - y, est[0] - x) - yaw)
            lead = abs(rate or 0.0) * p['scan_lead_s']
            if abs(err) > max(math.radians(p['show_face_deg']), lead) and t - self._show_t0 < 10.0:
                return Step(None, (0.0, math.copysign(p['scan_w'], err)), False, SHOW, 'facing the object')
        if self._show_phase == 'face':
            self._show_phase, self._show_t0 = 'hold', t
        if t - self._show_t0 < p['show_s']:
            return Step(None, (0.0, 0.0), False, SHOW, 'showing the object')
        self.state = ARRIVED
        note = f'{self.target.cls if self.target else "?"} shown'
        self.notes.append(f'{t:.1f} {note}')
        return Step(None, (0.0, 0.0), True, ARRIVED, note)
