"""Search mission with NO known map: build the 2D map from the free-space profile, explore it,
put every object found on it, and visit each one.

For "drive around indoors and find the potted plants" (object_detector_node, YOLO26). Where
loop_patrol.LoopPatrolMission patrols corridors it was given, this one has only what the rover
sees: the same log-odds grid local_planner.LocalCostmap builds from /erc/free_space, kept for
the whole mission in the mission frame (origin at the start pose), and the objects' positions
as markers on it (object_tracker.ObjectTracker: one track per object). No ROS:
image_checkpoint_controller_node (param explore) and test/explore_search_sim.py run this code,
through the same interface as LoopPatrolMission.

  EXPLORE  go to a frontier: free, seen ground next to ground the camera has not looked at,
           the rover can get to, chosen ahead of it when there is one near, by A* on the map. Reached, or no longer a frontier -> the next. None left near the
           rover -> look around once; none anywhere -> the area is explored.
  GOTO     an object is on the map (a confirmed track): A* to a standoff point short of
           it, the nearest unvisited first; the route is rebuilt as the estimate moves.
  SHOW     within arrive_m: face it, hold still show_s. ARRIVED -> the caller calls confirm().
  SCAN     turns in place: at the start, and when there is no frontier in reach.
  DONE     as many objects shown as there are goals (or patrol.max_shows), or nothing left to
           explore (stop_when_explored).

Driving past and counting (forward_first + count_on_sight, look_around off; what mission_plants
runs since 12-oct): the rover never turns to search and never goes to an object.
  * forward_first: while the profile shows an open way ahead that leads to ground not seen yet,
    the route is a straight line into the middle of that opening, rebuilt every second from
    the profile alone. The map is not asked: on the first real run two people walking up the
    corridor were written into it as a wall across the corridor, the route to a plant 5 m
    ahead went round the back (+149 deg), and the rover spent the run turning within 1.7 m of
    its start (rosbags/mission_1012_1429_macetas). Only with no such opening do the frontiers
    and A* above decide.
  * count_on_sight: an object counts when the rover comes within count_m of its track, as it
    drives by. Never one within same_class_apart_m of one already counted.
  * look_around off: with nowhere to go the rover stands still; after half of explored_after_s
    the map is started again (the rover may be boxed in by its own drifted map), and
    explored_after_s of standing ends the mission.

What it does not have, and what follows from that:
  * No drift correction. The map and the pose drift together, so what is near the rover is
    consistent with what it sees, but a place mapped long ago is displaced when the rover
    comes back to it. So the map forgets: a cell is dropped map_forget_m (25 m) of driving
    after it was last seen, and so are the objects not shown yet. Nothing here closes loops;
    an object shown before the rover drove a long loop can be shown again.
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
from erc_inference.loop_patrol import ARRIVED, DONE, GOTO, SCAN, SHOW, Step
from erc_inference.object_tracker import ObjectTracker, Track
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
    frontier_free_logodds=-0.79,    # ground counts as free for frontiers at this evidence: two free profiles
    frontier_spacing_m=1.5,     # one tried (reached or failed) is not picked again this close...
    frontier_memory_s=180.0,    # ...for this long
    goto_angles_deg=(0.0, 35.0, -35.0, 70.0, -70.0, 110.0, -110.0, 180.0),   # standoff points tried
                                # around the object, from the rover's side
    unreachable_s=60.0,         # an object with no path to any of them is left alone this long
    stop_when_explored=True,    # no frontier anywhere and no object pending: the mission ends
    route_clear_m=4.0,          # route length ahead checked against the map
    live_within_m=2.5,          # the approach follows the last live_s of sightings only this close
    arrive_lead_m=0.3,          # the stop is commanded this far before arrive_m (see update)
    map_forget_m=25.0,          # the map holds what was seen in the last this-many metres DRIVEN (0: all of
                                # it). The pose is wheels + gyro, nothing absolute: it drifts, and what was
                                # mapped long ago comes back displaced, a second wall beside the first.
                                # Rather than blanking the map every 25 m -- which would also blank what is
                                # right ahead -- each cell is dropped 25 m of driving after it was last seen.
    stuck_turning_deg=40.0,     # with the carrot this far off the heading the rover is turning in place
                                # (the carrot controller does from 45 deg), and not moving is expected: a
                                # frontier behind it takes > 12 s to face at that turn rate with the command
                                # delay, and read as "stuck" with nothing ahead (ROS e2e, 9-oct). Only for
    stuck_turn_grace_s=30.0,    # this long: a turn that never ends is being stuck too
    # driving past and counting (see the module docstring)
    count_on_sight=False,       # an object counts when the rover is within count_m of it; no GOTO, no SHOW
    count_m=2.0,
    count_min_sightings=8,      # ...and seen this many times: a track of 3-5 sightings is a piece of another,
                                # split off while the rover turned -- one plant was counted twice, 1.9 m apart,
                                # as #4 (25 sightings) and #5 (3) (ROS e2e with MPPI, 12-oct)
    forward_first=False,        # an open way ahead, leading to unseen ground, is taken before any frontier
    forward_fov_deg=40.0,       # looked for within this bearing (the map's limit too: the periphery is not trusted)
    forward_open_m=2.0,         # a bin is open when the profile reports nothing nearer
    forward_min_width_m=0.9,    # an opening is this wide at forward_open_m, or it is not one
    forward_clear_m=0.0,        # > 0: ...and nothing the whole profile reports within this of the straight line
                                # into it. OFF: tried at 0.35 because the rover rubbed a pot beside the opening
                                # (ROS e2e) -- it rubbed it all the same, and in explore_search_sim --count the
                                # seeds 1, 3 and 6 went from 5, 2, 9 touches to 19, 12, 9 and took twice as long
                                # (the map decides more often, and its routes turn more). 0.2 changed nothing.
    forward_reach_m=2.5,        # the straight route ends this far ahead, and 0.5 m short of what is seen there
    forward_every_s=1.0,        # rebuilt this often
    forward_grace_s=2.0,        # ...and kept this much longer when a profile shows no opening
    forward_max_turn_dps=20.0,  # not while the heading moves faster: the profile is ~0.4 s old
    forward_unseen_within_m=6.0,    # "leads to unseen ground": an unseen cell this near along it, before a wall
    forward_hold_s=15.0,        # after being stuck or tilted the map decides for this long (it holds the mark)
    forward_profile_age_s=1.5,
    look_around=True,           # False: no turning in place to look for a way on
    explored_after_s=20.0,      # look_around off: nowhere to go for this long = explored (people move away)
)

FORWARD = 'forward'             # a route kind, not a state: the state is EXPLORE


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
        self.cones = ObjectTracker()        # the objects, one track each; the attribute name is loop_patrol's
        self.odometer = 0.0                 # metres driven, for forgetting
        self._odo_xy = None
        self._forgot_at = 0.0
        self.index = 0
        self._arrive0 = self.p['arrive_m']
        self.notes: List[str] = []
        self.state = EXPLORE
        self.pts = self.cum = None
        self.s_proj = 0.0
        self._route_kind = None
        self._route_t = -math.inf
        self._route_to = None
        self.target: Optional[Track] = None
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
        self._profile_t = -math.inf
        self._fwd_dir = None             # (t, world angle) of the last forward route
        self._fwd_hold_until = -math.inf
        self._nothing_since = None       # since when there has been nowhere to go (look_around off)
        self._pick_t = -math.inf
        self._map_reset = False
        self._last_check = -math.inf
        self._backoff_until = None
        self._stuck_at = []              # (t, x, y) of recent stuck events
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

    def observe_objects(self, x, y, yaw, detections, t=None):
        """One camera frame's detections, [(class, bearing_deg, range_m), ...], seen from (x, y, yaw):
        all at once, so that two objects in one frame are never taken for one (object_tracker)."""
        if t is not None:
            self._yaws = [(tt, a) for tt, a in self._yaws if t - tt <= 1.0] + [(t, yaw)]
        if self.turn_rate_dps() > self.p['cone_max_turn_dps']:
            return []
        dets = [d for d in detections if d[2] <= self.p['max_detection_m']]
        return self.cones.observe(x, y, yaw, dets, 0.0 if t is None else t, self.odometer)

    def observe_cone(self, x, y, yaw, cls, bearing_deg, range_m, t=None):
        """One detection alone (loop_patrol's interface); a frame with several goes to observe_objects."""
        got = self.observe_objects(x, y, yaw, [(cls, bearing_deg, range_m)], t)
        return got[0] if got else None

    def observe_scan(self, t, x, y, yaw, bearings_deg, ranges, hit):
        b, r, h = np.asarray(bearings_deg, float), np.asarray(ranges, float), np.asarray(hit, bool)
        self._last_profile = (b, r, h)
        self._profile_t = t
        self.map.odometer = self.odometer
        self.replanner.observe(t, x, y, yaw, b, r, h)

    def confirm(self, t, accepted: bool):
        if self.target is not None:
            self.target.visited = True
            self.target.name = f'cone{self.index + 1}' if accepted else 'rejected'
            self._mark(self.target.estimate())
        self.notes.append(f'{t:.1f} {self.goal.name} ' + ('confirmed' if accepted else 'rejected'))
        if accepted:
            self.index += 1
        self.target = None
        self.state = EXPLORE if not self.done else DONE
        self._route_kind = None

    def _mark(self, xy, radius=0.3):
        """The object just shown is an obstacle where its track says: the rover stands 1-2 m from it
        about to drive on, and from that close the profile's rays often pass beside a pot (ROS e2e,
        12-oct: it rubbed the plant it had just shown, four times)."""
        m = self.map
        r = m.p['resolution_m']
        k = int(math.ceil(radius / r))
        r0, c0 = (int(v) for v in m.cell(xy[0], xy[1]))
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < m.h and 0 <= cc < m.w and math.hypot(dr, dc) * r <= radius:
                    m.L[rr, cc] = m.p['max_logodds']
                    m.seen[rr, cc] = True
                    m.travel[rr, cc] = self.odometer
        m._dist = None

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
        # Stuck again in the same place: what holds the rover is beside it, where the profile does not
        # look (a door jamb), so "nothing seen ahead" and the same route again, for ever (ROS e2e with
        # the turn gain measured at low speed, 12-oct: 17 times in 4 minutes). The second time, mark
        # ahead regardless; the third, give up on where it was going and go somewhere else.
        self._stuck_at = [e for e in self._stuck_at if t - e[0] <= p['stuck_memory_s']]
        here = sum(math.hypot(e[1] - x, e[2] - y) <= p['stuck_same_m'] for e in self._stuck_at)
        self._stuck_at.append((t, x, y))
        self._fwd_hold_until = t + p['forward_hold_s']
        mark = why.startswith('tilt') or here >= 1 or (ahead is not None and ahead < p['stuck_mark_to_m'] + 0.1)
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
            m.travel[rows[ok], cols[ok]] = self.odometer
            m._dist = None
        if self.frontier is not None:
            self._tried.append((t, *self.frontier))
            self.frontier = None
        gave_up = ''
        if here >= p['stuck_flip_after'] - 1 and self.target is not None and self.state != SHOW:
            self._skip[self.target.id] = t + 2.0 * p['unreachable_s']
            gave_up = f'; stuck here {here + 1} times: leaving the {self.target.cls} for later'
            self.target = None
            self.state = EXPLORE
        self._route_kind = None
        self._trail = []
        self.notes.append(f'{t:.1f} {why}: backing off, ' + ('obstacle marked ahead' if mark else 'nothing seen ahead')
                          + ', rerouting' + gave_up)

    def export_state(self):
        return {'index': self.index,
                'cones': [{'cls': c.cls, 'name': c.name, 'visited': c.visited,
                           'points': [[float(v) for v in c.estimate()]]}
                          for c in self.cones.clusters if c.confirmed]}

    def import_state(self, state, first_goal_index=None):
        """The objects and the count survive a restart of the node; the map is built again."""
        self.index = min(int(state.get('index', 0)), self.n_shows)
        self.cones = ObjectTracker()
        for c in state.get('cones', []):
            self.cones.add_known(c['cls'], np.median(np.array(c['points'], float), axis=0), c.get('name', ''),
                                 bool(c.get('visited', False)))
        self.target = None
        self.state = EXPLORE if not self.done else DONE
        self._route_kind = None
        self.notes.append(f'resumed at {self.index + 1}/{self.n_shows} with {len(self.cones.clusters)} objects mapped')

    # -- for looking at (the node publishes these) -----------------------------

    def objects(self):
        """[(class, x, y, visited, detections)] of every confirmed object on the map."""
        return [(f'{c.cls} #{c.id}', float(c.estimate()[0]), float(c.estimate()[1]),
                 bool(c.visited and c.name != 'rejected'), len(c.points)) for c in self.cones.clusters if c.confirmed]

    def _travel(self, x, y):
        """Metres driven; every metre, forget what was last seen map_forget_m ago and keep the grid
        around the rover."""
        if self._odo_xy is not None:
            d = math.hypot(x - self._odo_xy[0], y - self._odo_xy[1])
            if d < 1.0:                      # a jump is a new anchor, not driving
                self.odometer += d
        self._odo_xy = (x, y)
        if self.odometer - self._forgot_at < 1.0:
            return
        self._forgot_at = self.odometer
        m = self.map
        half = 0.5 * m.w * m.p['resolution_m']
        if max(abs(x - (m.ox + half)), abs(y - (m.oy + half))) > 0.5 * half:
            m.recenter(x, y)
        forget = float(self.p['map_forget_m'])
        if forget > 0.0:
            m.forget(self.odometer - forget)
            keep = self.target
            self.cones.forget(self.odometer - forget)
            if keep is not None and keep not in self.cones.clusters:
                self.cones.clusters.append(keep)     # the one it is going to stays

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

    def _first_leg_deg(self, x, y, yaw):
        """Where the route heads at the carrot, in degrees left of the rover's heading."""
        c = point_at(self.pts, self.cum, self.s_proj + self.p['carrot_distance_m'])
        return math.degrees(_wrap(math.atan2(c[1] - y, c[0] - x) - yaw))

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
        # free on two profiles at least: a ray that slips past a wall leaves its cells free ONCE, and
        # while that wall has been hit only once it is not yet a wall in the map. The frontier behind
        # it was then "reachable": 2.9 m, through the south wall of the first corridor, and with the
        # rover committed to it (_committed) it drove at the wall (ROS e2e with MPPI, 12-oct).
        passable = m.seen & (m.L <= self.p['frontier_free_logodds']) & (m.distance() > m.p['inflate_m'])
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

    # -- driving past and counting -------------------------------------------------

    def _opening(self, yaw):
        """The open way ahead in the last profile: (bearing_deg of its middle, free range there),
        or None. An opening is a run of bins within forward_fov_deg that all report nothing nearer
        than forward_open_m, at least forward_min_width_m wide at that distance; of several, the
        one nearest the direction the rover was already taking (straight ahead at first)."""
        p = self.p
        b_all, r_all, h_all = self._last_profile
        k = np.abs(b_all) <= p['forward_fov_deg']
        if k.sum() < 2:
            return None
        order = np.argsort(b_all[k])
        b, r = b_all[k][order], r_all[k][order]
        step = float(np.median(np.diff(b)))
        need = math.degrees(2.0 * math.atan(0.5 * p['forward_min_width_m'] / p['forward_open_m']))
        want = 0.0 if self._fwd_dir is None else math.degrees(_wrap(self._fwd_dir[1] - yaw))

        def clear(aim, reach):
            # nothing the WHOLE profile reports within forward_clear_m of the straight line (a pot
            # just outside the opening, 0.8 m away at 14 deg, is 0.2 m from it). See forward_clear_m.
            d = np.radians(b_all - aim)
            along, across = r_all * np.cos(d), np.abs(r_all * np.sin(d))
            return not (h_all & (along > 0.0) & (along < reach + 0.3) & (across < p['forward_clear_m'])).any()

        best = None
        i, n = 0, len(b)
        while i < n:
            if r[i] < p['forward_open_m']:
                i += 1
                continue
            j = i
            while j + 1 < n and r[j + 1] >= p['forward_open_m']:
                j += 1
            if b[j] - b[i] + step >= need:
                mid = 0.5 * (b[i] + b[j])
                reach = max(1.0, min(p['forward_reach_m'], float(r[i:j + 1].min()) - 0.5))
                # the middle of the opening, or the nearest direction to it that is clear and still
                # has half the needed width to each edge
                lo, hi = b[i] - 0.5 * step + 0.5 * need, b[j] + 0.5 * step - 0.5 * need
                cands = sorted([mid] + [float(c) for c in b[i:j + 1] if lo <= c <= hi], key=lambda c: abs(c - mid))
                aim = next((c for c in cands if clear(c, reach)), None)
                if aim is not None and (best is None or abs(aim - want) < abs(best[0] - want)):
                    best = (float(aim), float(r[i:j + 1].min()))
            i = j + 1
        return best

    def _leads_to_unseen(self, x, y, ang, clear_to):
        """True when, going from (x, y) along ang, ground the camera has not looked at comes before
        a wall of the map. What the profile shows free right now (clear_to) is not asked of the map:
        a person who walked through is still in it for a couple of seconds."""
        m = self.map
        d = np.arange(0.4, self.p['forward_unseen_within_m'] + 1e-9, 0.5 * m.p['resolution_m'])
        rows, cols = m.cell(x + d * math.cos(ang), y + d * math.sin(ang))
        ok = m._inside(rows, cols)
        if not ok.any():
            return False
        d, rows, cols = d[ok], rows[ok], cols[ok]
        unseen = ~m.seen[rows, cols]
        wall = (m.L[rows, cols] >= m.p['occupied_logodds']) & (d > clear_to)
        if not unseen.any():
            return False
        first = int(np.argmax(unseen))
        return not wall[:first].any()

    def _committed(self, x, y):
        """True while the rover is on its way to a frontier the map chose and that frontier is still
        one. The way ahead does not take over then: with both deciding, each for a second or two --
        an opening that leads to a pocket of unseen ground, a frontier 12 m back at -95 deg, the
        opening again -- the rover turned one way and the other against a wall (explore_search_sim
        --count seed 2: 1100 touches in 25 minutes)."""
        f = self.frontier
        if f is None or self._route_kind != EXPLORE:
            return False
        if math.hypot(f[0] - x, f[1] - y) <= self.p['frontier_reach_m'] or float(self.cum[-1]) - self.s_proj < 0.3:
            return False
        r, c = (int(v) for v in self.map.cell(f[0], f[1]))
        return not self.map.seen[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].all()

    def _forward(self, t, x, y, yaw):
        """Keep, or rebuild, the straight route into the open way ahead. -> True while there is one."""
        p = self.p
        if t < self._fwd_hold_until or self._last_profile is None:
            return False
        if self._route_kind == FORWARD and t - self._route_t < p['forward_every_s']:
            return True
        # One profile without the opening does not end it: the route in force was drawn over ground
        # seen free, and forward_grace_s of it is half a metre.
        keep = (self._route_kind == FORWARD and t - self._route_t < p['forward_every_s'] + p['forward_grace_s']
                and float(self.cum[-1]) - self.s_proj > 0.8)
        if t - self._profile_t > p['forward_profile_age_s']:
            return False
        if self.turn_rate_dps() > p['forward_max_turn_dps']:
            return self._route_kind == FORWARD      # mid-turn: the route it has, until the profile can be used
        got = self._opening(yaw)
        if got is None:
            return keep
        ang = yaw + math.radians(got[0])
        if self._fwd_dir is not None and t - self._fwd_dir[0] < 3.0:
            turn = _wrap(ang - self._fwd_dir[1])
            if abs(turn) < math.radians(40.0):          # half-way each second: one odd profile does not swing it
                ang = self._fwd_dir[1] + 0.5 * turn
        if not self._leads_to_unseen(x, y, ang, got[1]):
            return keep
        reach = max(1.0, min(p['forward_reach_m'], got[1] - 0.5))
        if self._route_kind != FORWARD:
            self.notes.append(f'{t:.1f} open ahead ({got[0]:+.0f} deg, {got[1]:.1f} m): driving on')
        self._fwd_dir = (t, ang)
        self.frontier = None
        self._set_route([(x, y), (x + reach * math.cos(ang), y + reach * math.sin(ang))], FORWARD, t)
        return True

    def _count(self, t, x, y):
        """count_on_sight: every confirmed object the rover is now within count_m of counts, once."""
        p = self.p
        for c in list(self.cones.clusters):
            if c.visited or not c.confirmed or self.index >= self.n_shows or len(c.points) < p['count_min_sightings']:
                continue
            e = c.live(t, p['live_s'])
            e = c.estimate() if e is None else e
            d = math.hypot(e[0] - x, e[1] - y)
            if d > p['count_m']:
                continue
            dup = next((v for v in self.cones.clusters if v is not c and v.visited and v.name.startswith('cone')
                        and v.cls == c.cls
                        and float(np.hypot(*(v.estimate() - c.estimate()))) <= p['same_class_apart_m']), None)
            c.visited = True
            if dup is not None:
                c.name = 'duplicate'
                self.notes.append(f'{t:.1f} the {c.cls} #{c.id} at ({e[0]:.1f},{e[1]:.1f}) is #{dup.id}, '
                                  f'already counted: not again')
                continue
            self.index += 1
            c.name = f'cone{self.index}'
            self.notes.append(f'{t:.1f} {c.cls} {self.index}/{self.n_shows} counted: #{c.id} at ({e[0]:.1f},{e[1]:.1f}), '
                              f'{d:.1f} m away, {len(c.points)} sightings')

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
        self._travel(x, y)
        if self.target is not None and self.target not in self.cones.clusters:
            self.target = None               # merged into another track, or forgotten
            self._route_kind = None
            if self.state in (SHOW, ARRIVED):
                # ...while it was being shown: nothing was shown. Left as it was, the hold ran out,
                # the caller confirmed "it" and one more object was counted with no object there
                # (explore_search_sim seed 2 crashed on it; the node would have counted it)
                self.state = EXPLORE
        if p['count_on_sight']:
            self._count(t, x, y)
            if self.done:
                self.notes.append(f'{t:.1f} {self.index}/{self.n_shows} objects counted')
                return Step(None, (0.0, 0.0), False, DONE, 'mission complete')
        elif self.target is None:
            shown = [c for c in self.cones.clusters if c.visited and c.name.startswith('cone')]

            def fresh(c):
                e = c.estimate()
                return all(v.cls != c.cls or float(np.hypot(*(v.estimate() - e))) > p['same_class_apart_m']
                           for v in shown)
            cands = [c for c in self.cones.clusters if not c.visited and c.confirmed
                     and fresh(c) and t >= self._skip.get(c.id, -math.inf)]
            if cands:
                self.target = min(cands, key=lambda c: float(np.hypot(*(c.estimate() - (x, y)))))
                e = self.target.estimate()
                self.notes.append(f'{t:.1f} object {self.index + 1}/{self.n_shows}: {self.target.cls} at '
                                  f'({e[0]:.1f},{e[1]:.1f})')
                self._route_kind = None
        est = None
        if self.target is not None:
            # the same object as one already shown, now that it is seen from closer: not shown twice
            e = self.target.estimate()
            dup = next((v for v in self.cones.clusters if v is not self.target and v.visited
                        and v.name.startswith('cone') and v.cls == self.target.cls
                        and float(np.hypot(*(v.estimate() - e))) <= p['same_class_apart_m']), None)
            if dup is not None:
                self.notes.append(f'{t:.1f} the {self.target.cls} #{self.target.id} at ({e[0]:.1f},{e[1]:.1f}) is #{dup.id}, '
                                  f'already shown: not again')
                self.target.visited, self.target.name = True, 'duplicate'
                self.target = None
                self._route_kind = None
        if self.target is None and self.state in (SHOW, ARRIVED):
            self.state = EXPLORE             # what was being shown turned out to be one already shown
        if self.target is not None:
            # Far away the position is every sighting fused; the last few seconds alone (a handful of
            # ranges, each good to ~12%) move it by a metre and the route with it. Close up, what is
            # in view now is the better guide.
            est = self.target.estimate()
            if math.hypot(est[0] - x, est[1] - y) <= p['live_within_m']:
                live = self.target.live(t, p['live_s'])
                if live is not None:
                    est = live
        if self.state == ARRIVED:
            return Step(None, (0.0, 0.0), True, ARRIVED)
        if self.state == SHOW:
            return self._show(t, x, y, yaw, est, rate)
        # arrive_lead_m sooner: the stop takes effect ~1 s later, 0.25-0.6 m further on. With the stop
        # commanded AT arrive_m 1.0 the rover came to rest 0.3-0.8 m from the plant, against the pot
        # more than once (ROS e2e, 12-oct).
        if est is not None and math.hypot(est[0] - x, est[1] - y) <= p['arrive_m'] + p['arrive_lead_m']:
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
            why = ('new' if self._route_kind != GOTO else
                   f'object moved {float(np.hypot(*(est - self._route_to))):.1f} m'
                   if np.hypot(*(est - self._route_to)) > p['reroute_moved_m'] else
                   'blocked' if self._route_blocked(t) else
                   'timer' if t - self._route_t > p['reroute_every_s'] else '')
            if why:
                before = self._first_leg_deg(x, y, yaw) if self._route_kind == GOTO else None
                ok = self._route_to_object(t, x, y, est)
                if ok:
                    after = self._first_leg_deg(x, y, yaw)
                    turned = None if before is None else abs((after - before + 180.0) % 360.0 - 180.0)
                    if why != 'timer' or (turned is not None and turned > 30.0):
                        self.notes.append(f'{t:.1f} route to #{self.target.id} ({why}): {float(self.cum[-1]):.1f} m, '
                                          f'starts {after:+.0f} deg from the heading'
                                          + (f' (was {before:+.0f})' if before is not None else ''))
                if not ok:
                    self._skip[self.target.id] = t + p['unreachable_s']
                    self.notes.append(f'{t:.1f} no path to the {self.target.cls} at ({est[0]:.1f},{est[1]:.1f}); '
                                      f'exploring on')
                    self.target = None
                    self.state = EXPLORE
                    self._route_kind = None
                    return Step(None, (0.0, 0.0), False, EXPLORE, 'no path to the object')
        elif p['forward_first'] and not self._committed(x, y) and self._forward(t, x, y, yaw):
            self.state = EXPLORE
            self._nothing_since = None
        else:
            self.state = EXPLORE
            if self._route_kind == FORWARD:      # the way ahead closed: the map decides
                self._route_kind = None
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
                if self._nothing_since is not None and t - self._pick_t < 1.0:
                    return Step(None, (0.0, 0.0), False, EXPLORE, 'no way on')     # A* once a second, not every tick
                self._pick_t = t
                if self._pick_frontier(t, x, y, yaw):
                    self._nothing_since = None
                    self.notes.append(f'{t:.1f} frontier at ({self.frontier[0]:.1f},{self.frontier[1]:.1f}): '
                                      f'{float(self.cum[-1]):.1f} m, starts {self._first_leg_deg(x, y, yaw):+.0f} deg '
                                      f'from the heading')
                elif not p['look_around']:
                    # Nowhere to go from what has been seen, and no turning to look for more: stand
                    # still. Whoever stands in the way may move; half-way through, the frontiers
                    # given up are tried again; explored_after_s of this is "explored".
                    if self._nothing_since is None:
                        self._nothing_since = t
                        self._map_reset = False
                        self.notes.append(f'{t:.1f} no way on from here: waiting')
                    waited = t - self._nothing_since
                    if not self._map_reset and waited > 0.5 * p['explored_after_s']:
                        # Half-way through: the map is started again. With frontiers on it and a path to
                        # none of them, the rover is standing inside a wall of its own map -- one drawn
                        # on the way out, the pose having drifted since (explore_search_sim --count
                        # seed 0: 41 frontiers, no path to any, mission ended with half the floor unseen).
                        # What it has counted stays counted.
                        self._map_reset = True
                        self._tried = []
                        m = self.map
                        m.L[:] = 0.0
                        m.seen[:] = False
                        m.travel[:] = -np.inf
                        m._dist = None
                        self._fwd_hold_until = -math.inf
                        self.notes.append(f'{t:.1f} still no way on: the map is started again')
                    if waited >= p['explored_after_s'] and self.replanner.scans > 0 and p['stop_when_explored']:
                        self.explored = True
                        self.notes.append(f'{t:.1f} nothing left to explore: {self.index}/{self.n_shows} objects')
                        return Step(None, (0.0, 0.0), False, DONE, 'explored')
                    return Step(None, (0.0, 0.0), False, EXPLORE, 'no way on')
                else:
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
