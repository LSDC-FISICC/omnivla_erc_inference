"""One leg of a cone mission: drive to the next coloured cone, replanning around obstacles.

For the NYU indoor track (and any image-goal mission whose checkpoints are
cones). No ROS: indoor_mission_node and test/cone_sim.py run this same code.

A leg starts with a PRIOR: the route from the track file (corridor corners)
to where the map says the cone is, about 1 m off, in a frame that drifts with
the odometry. What the leg does with the camera:

  cone      erc_perception.cones detections of the leg's colour (range from the
            cone's size, bearing from the base pixel), turned into track-frame
            points with the current pose and fused (median of the last fuse_n).
            The first one must fall within first_gate_m of the prior; later ones
            within gate_m of the estimate. Once min_confirm are in, the route is
            replaced by a straight line from the rover to a STANDOFF point
            standoff_m short of the cone, and rebuilt whenever the estimate moves
            more than reroute_moved_m. Detection reaches ~10 m, so for most of a
            28 m corridor the rover drives on the prior.
  obstacles /erc/free_space into local_planner.LocalReplanner (the one
            checkpoint_controller_node runs outdoors). When the route ahead is
            blocked it plans A* around the blockage and splices the detour in.
            The cone itself is 23 cm tall and would be mapped as an obstacle, so
            profile bins that look at the target cone are masked before mapping.
            That is also why the route ends at the standoff, not at the cone.
  look      optional stop-and-look (look_s): the first time the route is found
            blocked, hold still and keep integrating profiles before accepting a
            detour. The profile is 0.4-1.2 s old (image lag, see
            project notes); stationary, the lag does not smear the map.
  search    at the end of the prior route with no sighting: turn in place for
            up to search_s looking for the cone, then accept the prior and let
            the SDK decide.

Arrival: the fused estimate within arrive_m of the rover, or the prior (if the
cone was never seen) at the end of a search.

update() -> Step(carrot, arrived, command, note). command is None to let the
controller drive the carrot, or (v, w) to override it (hold still, search turn).
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from erc_inference import local_planner as lp

DEFAULTS = dict(
    standoff_m=1.0,
    arrive_m=1.5,
    fuse_n=7,
    first_gate_m=4.0,
    gate_m=1.5,
    min_confirm=2,
    reroute_moved_m=0.5,
    carrot_distance_m=1.5,
    cone_mask_deg=8.0,          # half-width of the masked sector around the target cone
    cone_mask_before_m=0.5,     # hits this much short of the cone are still mapped
    look_enabled=True,
    look_s=1.5,
    look_repeat_s=10.0,         # do not stop again for this long after a look
    search_enabled=True,
    search_w=0.3,               # rad/s in place (turns ~1.18x that in place)
    search_s=18.0,              # ~360 deg at 0.3 x 1.18 rad/s
    prior_arrive_m=1.5,         # end of the prior route
    pass_offset_m=0.6,          # pass a known cone (the one just reached) this far to its side:
                                # cone 0.09 + half footprint 0.125 + margin
)

# Same geometry as checkpoint_controller_node, on (x, y) = (east, north).
PROJECTION_WINDOW_M = 10.0


def _cumlen(pts):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])


def project_forward(pts, cum, x, y, s_min, window_m=PROJECTION_WINDOW_M):
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


class ConeTarget:
    """Track-frame estimate of one cone from repeated detections."""

    def __init__(self, prior_xy, p):
        self.prior = np.asarray(prior_xy, float)
        self.p = p
        self.points: List[np.ndarray] = []
        self.rejected = 0

    @property
    def confirmed(self):
        return len(self.points) >= self.p['min_confirm']

    def estimate(self) -> Optional[np.ndarray]:
        if not self.points:
            return None
        return np.median(np.array(self.points[-self.p['fuse_n']:]), axis=0)

    def add(self, x, y, yaw, bearing_deg, range_m) -> bool:
        a = yaw + math.radians(bearing_deg)
        pt = np.array([x + range_m * math.cos(a), y + range_m * math.sin(a)])
        ref, gate = (self.estimate(), self.p['gate_m']) if self.points else (self.prior, self.p['first_gate_m'])
        if np.hypot(*(pt - ref)) > gate:
            self.rejected += 1
            return False
        self.points.append(pt)
        return True


@dataclass
class Step:
    carrot: Tuple[float, float, float]
    arrived: bool
    command: Optional[Tuple[float, float]] = None
    note: str = ''
    route: Optional[np.ndarray] = field(default=None, repr=False)


class ConeLeg:
    def __init__(self, prior_route, cone_prior, color, lp_params=None, known_cones=(), **kw):
        """known_cones: track-frame (x, y) of cones already reached. The leg starts
        standing in front of the last one, and its route runs on through it."""
        self.p = dict(DEFAULTS)
        self.p.update(kw)
        self.color = color
        self.known_cones = [np.asarray(c, float) for c in known_cones]
        prior_route = self._pass_beside(np.asarray(prior_route, float), self.known_cones)
        self.pts = np.asarray(prior_route, float)
        self.cum = _cumlen(self.pts)
        self.s_proj = 0.0
        self.target = ConeTarget(cone_prior, self.p)
        box = np.vstack([self.pts, np.asarray(cone_prior, float)[None, :]])
        self.replanner = lp.LocalReplanner(box, **(lp_params or {}))
        m = self.replanner.map
        for c in self.known_cones:          # a cone we stood in front of is certain
            rows, cols = m.cell([c[0]], [c[1]])
            if m._inside(rows, cols).all():
                m.L[rows, cols] = m.p['max_logodds']
                m.seen[rows, cols] = True
                m._dist = None
        self._route_to = None           # cone estimate the current straight route was built for
        self._look_until = None
        self._look_done_t = -math.inf
        self._search_t0 = None
        self.notes: List[str] = []
        self.looks = 0
        self.searched = False

    def _pass_beside(self, route, cones):
        """Insert a point pass_offset_m beside any known cone the route's first
        segment runs over, on the rover's side of it."""
        if len(route) < 2 or not cones:
            return route
        a, b = route[0], route[1]
        ab = b - a
        L = float(np.hypot(*ab))
        if L < 1e-6:
            return route
        u = ab / L
        n = np.array([-u[1], u[0]])
        for c in cones:
            s_c = float((c - a) @ u)
            lateral = float((c - a) @ n)
            if 0.0 < s_c < L and abs(lateral) < self.p['pass_offset_m']:
                side = -1.0 if lateral > 0 else 1.0      # away from the cone
                via = c + side * self.p['pass_offset_m'] * n
                return np.vstack([route[:1], via[None, :], route[1:]])
        return route

    # -- inputs -------------------------------------------------------------

    def observe_cone(self, x, y, yaw, bearing_deg, range_m) -> bool:
        return self.target.add(x, y, yaw, bearing_deg, range_m)

    def observe_scan(self, t, x, y, yaw, bearings_deg, ranges, hit) -> bool:
        b = np.asarray(bearings_deg, float)
        r = np.asarray(ranges, float).copy()
        h = np.asarray(hit, bool).copy()
        est = self.target.estimate() if self.target.confirmed else None
        if est is not None:
            d = float(np.hypot(*(est - (x, y))))
            brg = math.degrees(math.atan2(est[1] - y, est[0] - x) - yaw)
            brg = (brg + 180.0) % 360.0 - 180.0
            half = max(self.p['cone_mask_deg'], math.degrees(math.atan2(0.3, max(d, 0.3))))
            mask = (np.abs(b - brg) <= half) & (r >= d - self.p['cone_mask_before_m'])
            h[mask] = False
        return self.replanner.observe(t, x, y, yaw, b, r, h)

    # -- the leg ------------------------------------------------------------

    def _straight_route(self, x, y, est):
        v = est - np.array([x, y])
        d = float(np.hypot(*v))
        end = est - v / d * self.p['standoff_m'] if d > self.p['standoff_m'] else np.array([x, y])
        self.pts = np.array([[x, y], end]) if d > self.p['standoff_m'] + 0.05 else np.array([[x, y], [x, y] + 0.01 * v / max(d, 1e-6)])
        self.cum = _cumlen(self.pts)
        self.s_proj = 0.0
        self._route_to = est.copy()

    def update(self, t, x, y, yaw) -> Step:
        p = self.p
        est = self.target.estimate() if self.target.confirmed else None
        # arrived at the cone
        if est is not None and math.hypot(est[0] - x, est[1] - y) <= p['arrive_m']:
            return Step(self._carrot(), True, (0.0, 0.0), f'at {self.color} cone')
        # searching at the end of the prior route
        if self._search_t0 is not None:
            if est is not None:
                self._search_t0 = None
                self.notes.append(f'{t:.1f} search: found {self.color}')
            elif t - self._search_t0 < p['search_s']:
                return Step(self._carrot(), False, (0.0, p['search_w']), 'searching')
            else:
                self._search_t0 = None
                return Step(self._carrot(), True, (0.0, 0.0), f'{self.color} not seen; prior accepted')
        # hold still while looking
        if self._look_until is not None:
            if t < self._look_until:
                return Step(self._carrot(), False, (0.0, 0.0), 'looking')
            self._look_until = None
            self._look_done_t = t
            self.replanner._last_check = -math.inf      # replan now, on the settled map
            self.replanner._last_plan = -math.inf
        # the route: prior until the cone is confirmed, then a line to its standoff
        if est is not None and (self._route_to is None
                                or math.hypot(*(est - self._route_to)) > p['reroute_moved_m']):
            self._straight_route(x, y, est)
            self.notes.append(f'{t:.1f} route to {self.color} cone at ({est[0]:.1f},{est[1]:.1f})')
        self.s_proj = project_forward(self.pts, self.cum, x, y, self.s_proj)
        new, note = self.replanner.check(t, x, y, self.pts, self.cum, self.s_proj)
        if note:
            if (p['look_enabled'] and new is not None and t - self._look_done_t > p['look_repeat_s']):
                self._look_until = t + p['look_s']
                self.looks += 1
                self.notes.append(f'{t:.1f} look ({note})')
                return Step(self._carrot(), False, (0.0, 0.0), 'looking')
            self.notes.append(f'{t:.1f} {note}')
        if new is not None:
            self.pts, self.cum, self.s_proj = new, _cumlen(new), 0.0
        # end of the prior route without a sighting
        if est is None and (float(self.cum[-1]) - self.s_proj < p['prior_arrive_m']
                            or math.hypot(*(self.target.prior - (x, y))) < p['prior_arrive_m']):
            if p['search_enabled'] and not self.searched:
                self.searched = True
                self._search_t0 = t
                self.notes.append(f'{t:.1f} search for {self.color}')
                return Step(self._carrot(), False, (0.0, p['search_w']), 'searching')
            return Step(self._carrot(), True, (0.0, 0.0), f'{self.color} not seen; prior accepted')
        return Step(self._carrot(), False, None, '', self.pts)

    def _carrot(self):
        return point_at(self.pts, self.cum, self.s_proj + self.p['carrot_distance_m'])
