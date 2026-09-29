"""Image-goal cone missions on a KNOWN corridor loop: patrol it, map every cone, visit them in order.

NYU indoor, ERC 2026: the mission is the PDF's cone images in order (CP1 red,
CP2 blue, CP3 green, CP4 yellow, Finish orange). The corridor walls are fixed
(indoor_nyu_track.yaml `corridors` / `loop`); the cones move on the day, and
chairs and open doors may be added. So the rover does not explore blind: it
patrols the known ring, records every cone it sees (any colour), and visits
them in the mission's order, going round the loop again if needed. No ROS:
image_checkpoint_controller_node and test/loop_patrol_sim.py run this code.

  KnownMap      corridors (centerline + half width), the patrol ring, scan points.
                lethal() = outside every corridor by more than wall_tolerance_m:
                the planner's static layer, so detours around a chair stay inside
                the corridor.
  WallLocalizer the fix for odometry drift, which in simulation put the rover into a
                wall on every run (indoor_sim.py: 104/104). Free-space hits that fall
                near a known wall are residuals of the pose: their median offset along
                the corridor normal corrects the lateral position, and their slope
                along the corridor the heading. Gated (a chair is not a wall) and
                skipped near junctions, where the walls open.
  ConeMap       every detection, any colour, clustered in the track frame. The
                Start/Finish cone is known from the start (next to the rover).
  LoopPatrolMission
                PATROL  follow the ring in `direction`, scanning +-90 deg once per lap at
                        each scan point (corners, junctions, stubs).
                GOTO    the current goal's cone is on the cone map: route along the ring,
                        the shorter way round, then straight to a standoff point short of
                        the cone; rebuilt as the estimate or the rover moves.
                ARRIVED the cone within arrive_m. The caller asks the SDK and calls
                        confirm(); a rejection halves the radius.
                Obstacles: local_planner.LocalReplanner (known walls as extra_lethal)
                splices detours; local_recovery.BlockedRecovery looks and replans when
                blocked. Blocked with no path, repeatedly: take the other way round.

Poses: the mission works in the corrected track frame. The caller feeds it the
odometry-anchored pose through WallLocalizer.apply() and publishes that same
corrected pose to the controller (the pseudo-fix), so carrot and rover agree.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from erc_inference import local_planner as lp
from erc_inference import local_recovery as lr
from erc_inference.image_goal_mission import Goal, _cumlen, _wrap, point_at, project_forward

PATROL, GOTO, SCAN, ARRIVED, DONE = 'patrol', 'goto', 'scan', 'arrived', 'done'

DEFAULTS = dict(
    direction=1,                # +1: the ring's own order (CCW on NYU), -1: the other way
    patrol_ahead_m=25.0,        # patrol route length ahead of the rover, rebuilt as it goes
    patrol_rebuild_m=10.0,
    carrot_distance_m=1.5,
    # cones
    join_m=1.5,                 # a detection this close to a cluster of its class joins it
    min_confirm=2,
    fuse_n=9,
    start_cone_m=2.0,           # detections of the start cone's class this close to the start merge into it
    max_detection_m=10.0,
    standoff_m=1.0,
    arrive_m=1.5,
    reroute_moved_m=0.5,
    reroute_every_s=5.0,
    cone_mask_deg=8.0,
    cone_mask_before_m=0.5,
    # scans
    scan_near_m=1.2,            # at a scan point when this close to it
    scan_deg=90.0,              # each side
    scan_w=0.3,
    scan_lead_s=2.4,
    scan_tol_deg=10.0,
    scan_timeout_s=12.0,
    scan_settle_s=1.5,
    # the map
    wall_tolerance_m=0.4,       # known walls are lethal only this far outside the corridor (drift)
    flip_after_no_path=3,       # blocked with no path this many times (each >= flip_spacing_s
    flip_spacing_s=8.0,         #   after the previous, within flip_window_s) -> the other way round
    flip_window_s=90.0,
    cone_max_turn_dps=10.0,     # detections while the heading turns faster than this are dropped:
                                # the image is 0.4-1.2 s older than the heading, so a scan turning
                                # at ~20 deg/s places a cone 9 m away ~2.5 m off (simulation)
    look=False,                 # local_recovery's stop-and-look (costs ~17 s each); the map is known
    # The final approach follows the LIVE view of the target: detections of its class within
    # target_gate_m of it join it, and with >= 2 from the last live_s the target is their median.
    # The map position of a cone is only as good as the pose it was seen from and the pose now;
    # in a ROS end-to-end run the rover "arrived" 0.4 m from a correctly mapped cone while
    # really 2.5-3.4 m away (pose drift), and the SDK rejected it forever (28-sept).
    target_gate_m=4.0,
    live_s=3.0,
    keep_after_reject=3,        # a rejection keeps only the last few points, then scans again
    # Stuck: the priority is to keep going (a bump at 0.25 m/s costs nothing, a deadlock ends the
    # run). Commanded to drive but less than stuck_min_move_m in stuck_window_s -> back off,
    # mark what is ahead as an obstacle, reroute; stuck twice within stuck_same_m -> the other way.
    stuck_window_s=12.0,
    stuck_min_move_m=0.3,
    backoff_s=1.8,
    backoff_v=-0.25,            # the floor the rover moves at, backwards (~0.4 m after the delay)
    stuck_mark_from_m=0.3,      # cells this far ahead...
    stuck_mark_to_m=0.9,        # ...to this far, +-stuck_mark_half_m sideways, marked occupied
    stuck_mark_half_m=0.25,
    stuck_same_m=2.5,
    stuck_flip_after=3,         # stuck this many times near the same place -> the other way round
                                # (2 ping-ponged: every 180 deg turn added gyro-scale error)
    stuck_memory_s=180.0,
)

# What the node runs indoors, here so that test/loop_patrol_sim.py runs the same:
# safety_envelope over its defaults -- 0.25 m/s (the floor the rover moves at), reverse allowed
# for the back-off, tilt ON (on a flat floor tilt means climbing something: stop and back off),
# and a missing attitude topic never freezes the rover (nothing could be tested on it).
# The obstacle STOP is OFF: in loop_patrol_sim (28-sept, 36 thirty-minute missions: 2 cone
# layouts x tables / chairs / chairs+doors x 6 seeds) it was the main thing keeping the rover
# from finishing -- 74/180 cones and 5/36 missions with it alone, 158/180 and 27/36 with it plus
# the stuck recovery, 174/180 and 32/36 without it (the recovery on). The cost is ~3-8 contacts
# per mission at 0.25 m/s, which the team accepts; tipping over is what must not happen.
SAFETY_INDOOR = {'safety.max_linear_vel': 0.25, 'safety.stop_distance_m': 0.7,
                 'safety.obstacle_enabled': False, 'safety.tilt_enabled': True,
                 'safety.stop_if_attitude_stale': False, 'safety.max_reverse_vel': 0.25}
# The planner's margin: 0.45 m (outdoor) needs 0.9 m each side of a cone on the centreline of a
# 1.9 m corridor, which is not there (loop_patrol_sim, 28-sept). The rover is 0.25 m wide.
LOCAL_INDOOR = {'inflate_m': 0.35,
                # Unseen ground within inflate + unseen_extend of a seen wall is lethal to A*: a
                # rule for garden beds (a wall seen in part probably continues). In a 2 m corridor
                # every unseen cell is that close to a wall, so every detour through ground the
                # camera has not looked at yet was "no path" (ROS e2e, 28-sept). The walls are known.
                'unseen_extend_m': 0.1}


# ---------------------------------------------------------------------------
# the known map
# ---------------------------------------------------------------------------

@dataclass
class Corridor:
    name: str
    a: np.ndarray
    b: np.ndarray
    hw: float
    walls_reliable: bool = True     # straight walls at +-hw: usable for drift correction

    def __post_init__(self):
        self.a, self.b = np.asarray(self.a, float), np.asarray(self.b, float)
        d = self.b - self.a
        self.length = float(np.hypot(*d))
        self.u = d / self.length
        self.n = np.array([-self.u[1], self.u[0]])

    def local(self, x, y):
        """(along, lateral) of points, along from a, lateral left-positive."""
        dx, dy = np.asarray(x, float) - self.a[0], np.asarray(y, float) - self.a[1]
        return dx * self.u[0] + dy * self.u[1], dx * self.n[0] + dy * self.n[1]


class KnownMap:
    def __init__(self, corridors, loop, scan_points=(), walls=(), furniture=()):
        self.corridors = [c if isinstance(c, Corridor) else
                          Corridor(c['name'], c['from'], c['to'], c['half_width'], bool(c.get('walls_reliable', True)))
                          for c in corridors]
        self.ring = np.asarray(loop, float)
        self.ring_cum = _cumlen(self.ring)
        self.ring_len = float(self.ring_cum[-1])
        self.scan_points = [np.asarray(p, float) for p in scan_points]
        segs, rel = [], []
        for wl in walls:
            if isinstance(wl, dict):
                segs.append([wl['from'], wl['to']])
                rel.append(bool(wl.get('reliable', True)))
            else:
                segs.append(wl)
                rel.append(True)
        w = np.asarray(segs, float).reshape(-1, 2, 2) if segs else np.zeros((0, 2, 2))
        self.wall_a, self.wall_b = w[:, 0, :], w[:, 1, :]
        self.wall_reliable = np.asarray(rel, bool)
        self.furniture = [(np.asarray(f['at'], float), float(f.get('r', 0.35))) for f in furniture]
        d = self.wall_b - self.wall_a
        self.wall_len = np.hypot(d[:, 0], d[:, 1])
        self.wall_u = d / np.maximum(self.wall_len, 1e-9)[:, None]
        self.wall_n = np.stack([-self.wall_u[:, 1], self.wall_u[:, 0]], axis=1)

    @classmethod
    def from_yaml(cls, doc):
        return cls(doc['corridors'], doc['loop'], doc.get('scan_points', []), doc.get('walls', []),
                   doc.get('furniture', []))

    def nearest_wall(self, px, py):
        """For points (N,): index of the nearest wall segment, and the signed distance along
        its normal (inf where there are no walls)."""
        P = np.stack([np.asarray(px, float), np.asarray(py, float)], axis=1)
        if len(self.wall_a) == 0:
            return np.zeros(len(P), int), np.full(len(P), np.inf)
        v = P[:, None, :] - self.wall_a[None, :, :]                       # (N, W, 2)
        t = np.clip(np.einsum('nwk,wk->nw', v, self.wall_u), 0.0, self.wall_len[None, :])
        foot = self.wall_a[None, :, :] + t[..., None] * self.wall_u[None, :, :]
        dist = np.hypot(*(P[:, None, :] - foot).transpose(2, 0, 1))
        j = np.argmin(dist, axis=1)
        signed = np.einsum('nk,nk->n', P - self.wall_a[j], self.wall_n[j])
        # beyond a segment's end the normal distance is not the distance: flag those
        inside = (t[np.arange(len(P)), j] > 0.0) & (t[np.arange(len(P)), j] < self.wall_len[j])
        return j, np.where(inside, signed, np.inf)

    def bounds(self, margin=3.0):
        pts = np.vstack([np.vstack([c.a, c.b]) for c in self.corridors])
        hw = max(c.hw for c in self.corridors)
        return pts.min(axis=0) - hw - margin, pts.max(axis=0) + hw + margin

    def inside(self, x, y, margin=0.0):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.zeros(np.broadcast(x, y).shape, bool)
        for c in self.corridors:
            s, l = c.local(x, y)
            ok |= (s >= -c.hw - margin) & (s <= c.length + c.hw + margin) & (np.abs(l) <= c.hw + margin)
        return ok

    def lethal_fn(self, tolerance):
        return lambda xs, ys: ~self.inside(xs, ys, tolerance)

    def corridor_at(self, x, y):
        """The corridor whose interior holds (x, y), preferring the one whose centerline
        is nearest; None outside all. Also whether it is near a junction (another corridor
        within 1 m), where the walls open and residuals mean nothing."""
        best, best_l = None, math.inf
        n_in = 0
        for c in self.corridors:
            s, l = c.local(x, y)
            if -0.5 <= s <= c.length + 0.5 and abs(l) <= c.hw + 0.6:
                n_in += int(abs(l) <= c.hw + 1.0)
                if abs(l) < best_l:
                    best, best_l = c, abs(l)
        return best, n_in > 1

    # -- the ring -----------------------------------------------------------
    def ring_s(self, x, y):
        """Arc length of the ring point nearest (x, y) (global search)."""
        return project_forward(self.ring, self.ring_cum, x, y, 0.0, window_m=1e9)

    def ring_path(self, s0, length, direction):
        """Points along the ring from s0 for `length` metres in `direction`, wrapping."""
        L = self.ring_len
        n = max(2, int(math.ceil(length / 0.5)) + 1)
        ss = (s0 + direction * np.linspace(0.0, length, n)) % L
        xs = np.interp(ss, self.ring_cum, self.ring[:, 0])
        ys = np.interp(ss, self.ring_cum, self.ring[:, 1])
        pts = np.stack([xs, ys], axis=1)
        # keep the corners: resampling at 0.5 m already lands within 0.25 m of them
        keep = np.concatenate([[True], np.hypot(*np.diff(pts, axis=0).T) > 1e-3])
        return pts[keep]

    def ring_gap(self, s_from, s_to, direction):
        return ((s_to - s_from) * direction) % self.ring_len


# ---------------------------------------------------------------------------
# drift correction against the known walls
# ---------------------------------------------------------------------------

class WallLocalizer:
    """Correction (dx, dy, dyaw) applied on top of the odometry-anchored pose.

    With the map's `walls`: point-to-wall matching (one damped Gauss-Newton step of ICP
    per profile). Side walls fix the lateral position and heading; an END wall in view
    (approaching a corner or a T) fixes the position along the corridor, which side walls
    cannot -- 2% of wheel scale left the rover turning 2 m early at corners otherwise
    (loop_patrol_sim, 28-sept). The damping keeps a direction nothing constrains (along a
    straight corridor) from moving. Without walls: the lateral/heading fit on the corridor.
    """

    def _observe_icp(self, x, y, yaw, bearings_deg, ranges, hit):
        b = np.radians(np.asarray(bearings_deg, float))
        r = np.asarray(ranges, float)
        h = np.asarray(hit, bool) & (r <= self.max_range) & np.isfinite(r)
        if h.sum() < self.min_points + 1:
            return None
        px, py = x + r[h] * np.cos(yaw + b[h]), y + r[h] * np.sin(yaw + b[h])
        j, res = self.km.nearest_wall(px, py)
        n = self.km.wall_n[j]
        # a point is compared with its wall's face on the rover's side
        side = np.sign(np.einsum('nk,nk->n', np.stack([x - self.km.wall_a[j][:, 0], y - self.km.wall_a[j][:, 1]], 1), n))
        res = res * side
        n = n * side[:, None]
        # Jacobian of the residual w.r.t. (dx, dy, dtheta) about the rover
        vx, vy = px - x, py - y
        J = np.stack([n[:, 0], n[:, 1], n[:, 0] * -vy + n[:, 1] * vx], axis=1)

        def solve(k):
            A = J[k].T @ J[k] + np.diag([self.damp_xy, self.damp_xy, self.damp_yaw])
            return -np.linalg.solve(A, J[k].T @ res[k])

        # a hit whose nearest wall is unreliable (curved, partly mapped) is not matched at all,
        # and neither is one on known furniture (a table read as the wall 1 m closer)
        k = np.isfinite(res) & (np.abs(res) <= self.gate) & self.km.wall_reliable[j]
        for c, rad in self.km.furniture:
            k &= np.hypot(px - c[0], py - c[1]) > rad + 0.3
        if k.sum() < self.min_points:
            return None
        d = solve(k)
        k = k & (np.abs(res + J @ d) <= self.refit_gate)     # a chair, a door frame, a person
        if k.sum() < self.min_points:
            return None
        d = solve(k)
        sx, sy, a = self.gain_xy * d[0], self.gain_xy * d[1], self.gain_yaw * d[2]
        self.dx, self.dy = self.dx + sx, self.dy + sy
        if a != 0.0:
            cx, cy = x + sx, y + sy
            tx, ty = self.dx - cx, self.dy - cy
            ca, sa = math.cos(a), math.sin(a)
            self.dx, self.dy = ca * tx - sa * ty + cx, sa * tx + ca * ty + cy
            self.dyaw = _wrap(self.dyaw + a)
        self.updates += 1
        return float(d[0]), float(d[1]), float(d[2])

    def __init__(self, known: KnownMap, gain_xy=0.3, gain_yaw=0.15, gate_m=0.7, refit_gate_m=0.25,
                 max_range_m=2.5, min_points=3, min_span_m=0.8, slope_agree=0.05,
                 damp_xy=2.0, damp_yaw=4.0, max_turn_dps=10.0):
        self.km = known
        self.gain_xy, self.gain_yaw, self.gate, self.max_range = gain_xy, gain_yaw, gate_m, max_range_m
        self.refit_gate = refit_gate_m
        self.slope_agree = slope_agree
        self.damp_xy, self.damp_yaw = damp_xy, damp_yaw
        self.max_turn_dps = max_turn_dps
        self.min_points, self.min_span = min_points, min_span_m
        self.dx = self.dy = self.dyaw = 0.0
        self.updates = 0

    def apply(self, x, y, yaw):
        """Odometry-anchored pose -> corrected track pose (rotation about the origin, then shift)."""
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        return c * x - s * y + self.dx, s * x + c * y + self.dy, _wrap(yaw + self.dyaw)

    def observe(self, x, y, yaw, bearings_deg, ranges, hit, turn_rate_dps=0.0):
        """One profile seen from the CORRECTED pose. -> the correction applied, or None.
        Skipped while the heading turns faster than max_turn_dps: the profile is 0.4-1.2 s
        older than the heading, and a lagged profile matched mid-turn rotates the pose."""
        if turn_rate_dps > self.max_turn_dps:
            return None
        corr, junction = self.km.corridor_at(x, y)
        if corr is None:
            return None
        if len(self.km.wall_a):
            return self._observe_icp(x, y, yaw, bearings_deg, ranges, hit)
        if not corr.walls_reliable:
            return None
        if junction:
            return None
        b = np.radians(np.asarray(bearings_deg, float))
        r = np.asarray(ranges, float)
        h = np.asarray(hit, bool) & (r <= self.max_range) & np.isfinite(r)
        if h.sum() < self.min_points:
            return None
        px, py = x + r[h] * np.cos(yaw + b[h]), y + r[h] * np.sin(yaw + b[h])
        s, l = corr.local(px, py)
        inside = (s >= 0.0) & (s <= corr.length)
        side = np.sign(l)
        e = l - side * corr.hw                 # >0: the wall appears further left than it is
        good = inside & (np.abs(e) <= self.gate) & (side != 0)
        # hits in another corridor's mouth are not this corridor's walls
        for c in self.km.corridors:
            if c is corr:
                continue
            cs_, cl_ = c.local(px, py)
            good &= ~((cs_ >= -c.hw) & (cs_ <= c.length + c.hw) & (np.abs(cl_) <= c.hw + 0.3))
        if good.sum() < self.min_points:
            return None
        s_rover, _ = corr.local(x, y)

        def fit(k):
            ds = s[k] - s_rover
            if k.sum() >= self.min_points and np.ptp(ds) >= self.min_span:
                # e = lateral + tan(heading error) * distance along: a heading error rotates
                # every hit about the rover, so the residual grows with how far along it is
                A = np.vstack([np.ones(k.sum()), ds]).T
                (c0, c1), *_ = np.linalg.lstsq(A, e[k], rcond=None)
                return float(c0), float(c1)
            return float(np.median(e[k])), 0.0

        lat, slope = fit(good)
        # second pass, tight around the first fit: a chair near a wall must not pull the pose
        pred = lat + slope * (s - s_rover)
        tight = good & (np.abs(e - pred) <= self.refit_gate)
        if tight.sum() < self.min_points:
            return None
        lat, slope = fit(tight)
        # the heading only when both walls agree on it: an object against ONE wall (a chair)
        # bends that wall's residual and would rotate the pose (5 deg in simulation)
        side_slopes = []
        for sd in (-1.0, 1.0):
            k = tight & (side == sd)
            if k.sum() >= self.min_points and np.ptp(s[k]) >= self.min_span:
                side_slopes.append(fit(k)[1])
        if len(side_slopes) == 2 and abs(side_slopes[0] - side_slopes[1]) <= self.slope_agree:
            dyaw = -math.atan(0.5 * (side_slopes[0] + side_slopes[1]))
        else:
            dyaw = 0.0
            lat = float(np.median(e[tight] - slope * (s[tight] - s_rover))) if len(side_slopes) == 2 else lat
        # move the corrected pose back across the corridor by a share of the residual...
        sx, sy = -self.gain_xy * lat * corr.n[0], -self.gain_xy * lat * corr.n[1]
        self.dx, self.dy = self.dx + sx, self.dy + sy
        # ...and rotate the correction about the rover's corrected position by a share of the
        # heading error: C'(p) = R(a) (C(p) - c) + c
        a = self.gain_yaw * dyaw
        if a != 0.0:
            cx, cy = x + sx, y + sy
            tx, ty = self.dx - cx, self.dy - cy
            ca, sa = math.cos(a), math.sin(a)
            self.dx, self.dy = ca * tx - sa * ty + cx, sa * tx + ca * ty + cy
            self.dyaw = _wrap(self.dyaw + a)
        self.updates += 1
        return lat, dyaw


# ---------------------------------------------------------------------------
# the cone map
# ---------------------------------------------------------------------------

@dataclass
class ConeCluster:
    cls: str
    points: List[np.ndarray] = field(default_factory=list)
    name: str = ''              # the goal it was confirmed as
    visited: bool = False
    times: List[float] = field(default_factory=list)

    def estimate(self, n=9):
        return np.median(np.array(self.points[-n:]), axis=0)

    def add(self, xy, t):
        self.points.append(np.asarray(xy, float))
        self.times.append(-math.inf if t is None else float(t))

    def live(self, t, window):
        """Median of the detections from the last `window` s, or None with fewer than 2."""
        if t is None:
            return None
        k = [i for i, tt in enumerate(self.times) if t - tt <= window]
        if len(k) < 2:
            return None
        return np.median(np.array([self.points[i] for i in k]), axis=0)


class ConeMap:
    def __init__(self, p):
        self.p = p
        self.clusters: List[ConeCluster] = []

    def add(self, cls, xy, t=None, prefer: Optional['ConeCluster'] = None, prefer_gate=0.0) -> ConeCluster:
        xy = np.asarray(xy, float)
        start = next((c for c in self.clusters if c.name == 'start'), None)
        near_start = start is not None and start.cls == cls and \
            np.hypot(*(start.estimate(self.p['fuse_n']) - xy)) <= self.p['start_cone_m']
        if prefer is not None and prefer.cls == cls and prefer.points and not near_start \
                and np.hypot(*(prefer.estimate(self.p['fuse_n']) - xy)) <= prefer_gate:
            prefer.add(xy, t)          # the target follows the live view
            return prefer
        best, best_d = None, math.inf
        for c in self.clusters:
            if c.cls != cls:
                continue
            d = float(np.hypot(*(c.estimate(self.p['fuse_n']) - xy)))
            lim = self.p['start_cone_m'] if c.name == 'start' else self.p['join_m']
            if d <= lim and d < best_d:
                best, best_d = c, d
        if best is None:
            best = ConeCluster(cls)
            self.clusters.append(best)
        best.add(xy, t)
        return best

    def candidates(self, cls):
        return [c for c in self.clusters if c.cls == cls and not c.visited and c.name != 'start'
                and len(c.points) >= self.p['min_confirm']]


# ---------------------------------------------------------------------------
# the mission
# ---------------------------------------------------------------------------

@dataclass
class Step:
    carrot: Optional[Tuple[float, float, float]]
    command: Optional[Tuple[float, float]]
    arrived: bool
    state: str
    note: str = ''
    route: Optional[np.ndarray] = field(default=None, repr=False)


class LoopPatrolMission:
    def __init__(self, goals: Sequence[Goal], known: KnownMap, start_xy=(0.0, 0.0), start_cone_class='red_orange',
                 lp_params=None, recovery_params=None, **kw):
        self.p = dict(DEFAULTS)
        self.p.update(kw)
        self.goals = list(goals)
        self.km = known
        self.start = np.asarray(start_xy, float)
        lo, hi = known.bounds()
        lp_kw = dict(lp_params or {})
        lp_kw['margin_m'] = 1.0
        self.replanner = lp.LocalReplanner(np.array([lo, hi]), **lp_kw)
        rec = dict(look=self.p['look'], explore=False)
        rec.update(recovery_params or {})
        self.recovery = lr.BlockedRecovery(**rec)
        self.lethal = known.lethal_fn(self.p['wall_tolerance_m'])
        self.cones = ConeMap(self.p)
        start = ConeCluster(start_cone_class, [self.start.copy()] * self.p['min_confirm'], name='start',
                            times=[-math.inf] * self.p['min_confirm'])
        self.cones.clusters.append(start)
        self.direction = int(self.p['direction'])
        self.index = 0
        self._arrive0 = self.p['arrive_m']
        self.notes: List[str] = []
        self.state = PATROL
        self.pts = self.cum = None
        self.s_proj = 0.0
        self._s_ring = None
        self._route_kind = None
        self._route_t = -math.inf
        self._route_to = None
        self._last_dir = None
        self.target: Optional[ConeCluster] = None
        self._scanned = set()            # (lap, scan point index)
        self._lap = 0
        self._travel = 0.0
        self._last_xy = None
        self._scan_queue = []
        self._scan_target = None
        self._scan_phase = None
        self._scan_t0 = 0.0
        self._no_path = []
        self._yaws = []
        self._trail = []                 # (t, x, y) while driving the route
        self._last_profile = None
        self._backoff_until = None
        self._stuck_at = []              # (t, x, y, direction) of recent stuck events
        self._avoid_dir = None           # (direction, until): the way round that got stuck
        self.stucks = 0

    @property
    def goal(self) -> Optional[Goal]:
        return self.goals[self.index] if self.index < len(self.goals) else None

    @property
    def done(self):
        return self.index >= len(self.goals)

    # -- inputs -------------------------------------------------------------

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
        tgt = self.target if (self.target is not None and self.target.name != 'start') else None
        return self.cones.add(cls, (x + range_m * math.cos(a), y + range_m * math.sin(a)), t,
                              prefer=tgt, prefer_gate=self.p['target_gate_m'])

    def observe_scan(self, t, x, y, yaw, bearings_deg, ranges, hit):
        b = np.asarray(bearings_deg, float)
        r = np.asarray(ranges, float)
        h = np.asarray(hit, bool).copy()
        self._last_profile = (b, r, np.asarray(hit, bool))
        # only the TARGET cone is masked (the route ends short of it); every other cone is an
        # obstacle like any other, or the planner routes straight through it
        for c in ([self.target] if self.target is not None else []):
            if len(c.points) < self.p['min_confirm']:
                continue
            est = c.estimate(self.p['fuse_n'])
            d = float(np.hypot(*(est - (x, y))))
            if d > 4.0:
                continue
            brg = math.degrees(_wrap(math.atan2(est[1] - y, est[0] - x) - yaw))
            half = max(self.p['cone_mask_deg'], math.degrees(math.atan2(0.3, max(d, 0.3))))
            h[(np.abs(b - brg) <= half) & (r >= d - self.p['cone_mask_before_m'])] = False
        return self.replanner.observe(t, x, y, yaw, b, r, h)

    def confirm(self, t, accepted: bool):
        if accepted:
            if self.target is not None:
                self.target.visited = True
                self.target.name = self.goal.name if self.target.name != 'start' else 'start'
                self._mark(self.target.estimate(self.p['fuse_n']))
            self.notes.append(f'{t:.1f} {self.goal.name} confirmed')
            self.index += 1
            self.p['arrive_m'] = self._arrive0
            self.target = None
            self.state = PATROL if not self.done else DONE
            self._route_kind = None
        else:
            self.p['arrive_m'] = max(self.p['standoff_m'] + 0.1, 0.5 * self.p['arrive_m'])
            if self.target is not None and self.target.name != 'start':
                k = self.p['keep_after_reject']
                self.target.points = self.target.points[-k:]
                self.target.times = self.target.times[-k:]
            # look again: the cone's position (or the pose) was not what the SDK sees
            a = math.radians(self.p['scan_deg'])
            yaw = self._yaws[-1][1] if self._yaws else 0.0
            self._scan_queue, self._scan_target, self._scan_phase = [yaw + a, yaw - a, yaw], None, None
            self.state = SCAN
            self.notes.append(f'{t:.1f} rejected; looking for the cone again, closing in to {self.p["arrive_m"]:.1f} m')

    def block_ahead(self, t, x, y, yaw, why):
        """Something ahead stopped the rover (stuck, or tilting onto it): back off, mark it,
        reroute; the second time near the same place, take the other way round."""
        p = self.p
        self.stucks += 1
        self._backoff_until = t + p['backoff_s']
        # mark ahead only when the profile does show something close there: marking blind, every
        # stuck event filled a 2 m corridor a little more until no path was left (ROS e2e, 28-sept)
        ahead = self._ahead_m()
        mark = why.startswith('tilt') or (ahead is not None and ahead < p['stuck_mark_to_m'] + 0.1)
        m = self.replanner.map
        ds = np.arange(p['stuck_mark_from_m'], p['stuck_mark_to_m'] + 1e-9, m.p['resolution_m'] * 0.5)
        dl = np.arange(-p['stuck_mark_half_m'], p['stuck_mark_half_m'] + 1e-9, m.p['resolution_m'] * 0.5)
        S, Lg = np.meshgrid(ds, dl)
        xs = x + S * math.cos(yaw) - Lg * math.sin(yaw)
        ys = y + S * math.sin(yaw) + Lg * math.cos(yaw)
        if mark:
            rows, cols = m.cell(xs.ravel(), ys.ravel())
            ok = m._inside(rows, cols)
            m.L[rows[ok], cols[ok]] = m.p['max_logodds']
            m.seen[rows[ok], cols[ok]] = True
            m._dist = None
        self._stuck_at = [e for e in self._stuck_at if t - e[0] <= p['stuck_memory_s']]
        again = sum(math.hypot(e[1] - x, e[2] - y) <= p['stuck_same_m'] for e in self._stuck_at) >= p['stuck_flip_after'] - 1
        self._stuck_at.append((t, x, y, self.direction))
        if again:
            self._avoid_dir = (self._last_dir if self._last_dir is not None else self.direction, t + 120.0)
            if self.state == PATROL:
                self.direction = -self.direction
        self._route_kind = None
        self._trail = []
        self.notes.append(f'{t:.1f} {why}: backing off, ' + ('obstacle marked ahead' if mark else 'nothing seen ahead')
                          + ', rerouting'
                          + ('; stuck here before -> the other way round' if again else ''))

    def _ahead_m(self, sector_deg=20.0):
        if self._last_profile is None:
            return None
        b, r, h = self._last_profile
        k = (np.abs(b) <= sector_deg) & h
        return float(r[k].min()) if k.any() else None

    def _check_stuck(self, t, x, y):
        p = self.p
        self._trail = [e for e in self._trail if t - e[0] <= p['stuck_window_s']] + [(t, x, y)]
        if self._trail[-1][0] - self._trail[0][0] < p['stuck_window_s'] - 0.5:
            return False
        xs = np.array([e[1] for e in self._trail])
        ys = np.array([e[2] for e in self._trail])
        return float(np.hypot(xs - xs[0], ys - ys[0]).max()) < p['stuck_min_move_m']

    def _mark(self, xy):
        m = self.replanner.map
        rows, cols = m.cell([xy[0]], [xy[1]])
        if m._inside(rows, cols).all():
            m.L[rows, cols] = m.p['max_logodds']
            m.seen[rows, cols] = True
            m._dist = None

    # -- routes -------------------------------------------------------------

    def _set_route(self, pts, kind):
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            pts = np.vstack([pts, pts[-1:] + 0.01])
        self.pts, self.cum, self.s_proj = pts, _cumlen(pts), 0.0
        self._route_kind = kind

    def _patrol_route(self, t, x, y):
        s = self.km.ring_s(x, y) if self._s_ring is None else self._ring_s_near(x, y)
        path = self.km.ring_path(s, self.p['patrol_ahead_m'], self.direction)
        self._set_route(np.vstack([[x, y], path]), PATROL)
        self._route_t = t

    def _ring_s_near(self, x, y):
        """Ring arc length near the last one (the start spur is on the ring twice)."""
        L = self.km.ring_len
        ds = np.arange(-15.0, 15.01, 0.25)
        ss = (self._s_ring + ds) % L
        px = np.interp(ss, self.km.ring_cum, self.km.ring[:, 0])
        py = np.interp(ss, self.km.ring_cum, self.km.ring[:, 1])
        d = np.hypot(px - x, py - y) + 0.02 * np.abs(ds)
        return float(ss[int(np.argmin(d))])

    def _goto_route(self, t, x, y, est):
        s_r = self._ring_s_near(x, y) if self._s_ring is not None else self.km.ring_s(x, y)
        s_c = self.km.ring_s(*est)
        fwd = self.km.ring_gap(s_r, s_c, self.direction)
        back = self.km.ring_gap(s_r, s_c, -self.direction)
        d, gap = (self.direction, fwd) if fwd <= back else (-self.direction, back)
        if self._avoid_dir is not None and t < self._avoid_dir[1] and d == self._avoid_dir[0] and gap > 3.0:
            d, gap = (-d, back if d == self.direction else fwd)      # stuck that way: the other way round
        self._last_dir = d
        v = est - np.array([x, y])
        dist = float(np.hypot(*v))
        if gap < 3.0 or dist < 4.0:
            path = np.zeros((0, 2))
        else:
            path = self.km.ring_path(s_r, max(0.0, gap - 2.0), d)
        last = path[-1] if len(path) else np.array([x, y])
        w = est - last
        wd = float(np.hypot(*w))
        end = est - w / wd * self.p['standoff_m'] if wd > self.p['standoff_m'] + 0.05 else last
        pts = np.vstack([[x, y], path, end]) if len(path) else np.vstack([[x, y], end])
        self._set_route(pts, GOTO)
        self._route_t = t
        self._route_to = est.copy()
        return d, gap

    # -- scans --------------------------------------------------------------

    def _maybe_scan(self, t, x, y, yaw):
        for i, sp in enumerate(self.km.scan_points):
            key = (self._lap, i)
            if key in self._scanned or math.hypot(sp[0] - x, sp[1] - y) > self.p['scan_near_m']:
                continue
            self._scanned.add(key)
            a = math.radians(self.p['scan_deg'])
            self._scan_queue = [yaw + a, yaw - a, yaw]
            self._scan_target, self._scan_phase = None, None
            self.state = SCAN
            self.notes.append(f'{t:.1f} scan at ({sp[0]:.1f},{sp[1]:.1f})')
            return True
        return False

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

    # -- the step -----------------------------------------------------------

    def update(self, t, x, y, yaw, rate=0.0, reflex_idle=True) -> Step:
        p = self.p
        if self.done:
            return Step(None, (0.0, 0.0), False, DONE, 'mission complete')
        self._yaws = [(tt, a) for tt, a in self._yaws if t - tt <= 1.0] + [(t, yaw)]
        # laps: ring arc length travelled in the patrol direction
        s_now = self.km.ring_s(x, y) if self._s_ring is None else self._ring_s_near(x, y)
        if self._s_ring is not None:
            ds = ((s_now - self._s_ring + self.km.ring_len / 2) % self.km.ring_len) - self.km.ring_len / 2
            self._travel += ds * self.direction
            self._lap = max(self._lap, int(self._travel // self.km.ring_len))
        self._s_ring = s_now
        g = self.goal
        # the target: the goal's cone on the cone map (the start cone for the Finish)
        if self.target is None:
            if g.is_start_cone:
                self.target = next(c for c in self.cones.clusters if c.name == 'start')
            else:
                cands = self.cones.candidates(g.cone_class)
                if cands:
                    self.target = min(cands, key=lambda c: float(np.hypot(*(c.estimate() - (x, y)))))
                    self.notes.append(f'{t:.1f} {g.name}: cone on the map at '
                                      f'({self.target.estimate()[0]:.1f},{self.target.estimate()[1]:.1f})')
        est = None
        if self.target is not None:
            est = self.target.live(t, p['live_s'])
            if est is None:
                est = self.target.estimate(p['fuse_n'])
        if self.state == ARRIVED:
            return Step(None, (0.0, 0.0), True, ARRIVED)
        if est is not None and math.hypot(est[0] - x, est[1] - y) <= p['arrive_m']:
            self.state = ARRIVED
            note = f'{g.name}: {math.hypot(est[0] - x, est[1] - y):.1f} m from the cone'
            self.notes.append(f'{t:.1f} arrived {note}')
            return Step(None, (0.0, 0.0), True, ARRIVED, note)
        if self._backoff_until is not None:
            if t < self._backoff_until:
                return Step(None, (p['backoff_v'], 0.0), False, self.state, 'backing off')
            self._backoff_until = None
        if self.state == SCAN:
            cmd = self._scan_step(t, yaw, 0.0 if rate is None else rate)
            if cmd is not None:
                self._trail = []
                return Step(None, cmd, False, SCAN, 'scanning')
            self.state = PATROL
            self._route_kind = None
        if est is not None:
            self.state = GOTO
            if (self._route_kind != GOTO or np.hypot(*(est - self._route_to)) > p['reroute_moved_m']
                    or t - self._route_t > p['reroute_every_s']) and not self.recovery.active:
                d, gap = self._goto_route(t, x, y, est)
        else:
            self.state = PATROL
            if self._maybe_scan(t, x, y, yaw):
                return Step(None, (0.0, 0.0), False, SCAN, 'scan')
            if (self._route_kind != PATROL or float(self.cum[-1]) - self.s_proj < p['patrol_rebuild_m']) \
                    and not self.recovery.active:
                self._patrol_route(t, x, y)
        # drive the route: detours and recovery, as outdoors
        self.s_proj = project_forward(self.pts, self.cum, x, y, self.s_proj)
        goal_xy = est if est is not None else self.pts[-1]
        rnote = ''
        if not self.recovery.active:
            new, note = self.replanner.check(t, x, y, self.pts, self.cum, self.s_proj, self.lethal)
            if new is not None:
                self._set_route(new, self._route_kind)
                self._route_t = t
            if note.startswith('[local-plan] blocked'):
                gb = _wrap(math.atan2(goal_xy[1] - y, goal_xy[0] - x) - yaw)
                self.recovery.trigger(t, yaw, gb, reflex_idle=reflex_idle, no_path='no path' in note)
                if 'no path' in note:
                    self._no_path = [tt for tt in self._no_path if t - tt <= p['flip_window_s']]
                    if not self._no_path or t - self._no_path[-1] >= p['flip_spacing_s']:
                        self._no_path.append(t)
                    if len(self._no_path) >= p['flip_after_no_path']:
                        self.direction = -self.direction
                        self._no_path = []
                        self._route_kind = None
                        self.notes.append(f'{t:.1f} no way through: the other way round')
            if note:
                self.notes.append(f'{t:.1f} {note}')
        cmd, new2, rnote = self.recovery.step(t, x, y, yaw, self.replanner, self.pts, self.cum, self.s_proj,
                                              tuple(goal_xy), self.lethal)
        if new2 is not None:
            self._set_route(new2, self._route_kind)
            self._route_t = t
        if rnote:
            self.notes.append(f'{t:.1f} {rnote}')
        if self.recovery.active:
            self._trail = []
            return Step(self._carrot(), cmd if cmd is not None else (0.0, 0.0), False, self.state, 'recovering')
        if self._check_stuck(t, x, y):
            self.block_ahead(t, x, y, yaw, 'stuck')
            return Step(None, (p['backoff_v'], 0.0), False, self.state, 'backing off')
        return Step(self._carrot(), None, False, self.state, '', self.pts)

    def _carrot(self):
        return point_at(self.pts, self.cum, self.s_proj + self.p['carrot_distance_m'])
