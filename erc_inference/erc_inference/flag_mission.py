"""Off-road checkpoint flags: reach, in order, the blue flags the checkpoints are. No GPS, no map.

The organisers (29-sept): three blue flags serve as the checkpoints; get within 1 m of each.
The goal photos (any camera, e.g. a phone, portrait) show each flag in its place; the flags
themselves all look the same. No ROS: flag_checkpoint_node and test/homing_sim.py run this.

  map      every flag seen (erc_perception.flags: bearing + range from the cloth's apparent
           height) is a cluster in the odometry frame. Its position and the cloth's real height
           are fitted together from odometry (least squares over bearing + angular height), so
           the assumed flag_height_m only matters until the rover has moved towards it.
  which    for checkpoint k: the unvisited flag seen in the frames that best match goal photo k
           (SIFT + fundamental matrix: no camera model needed), else the nearest unvisited flag.
  drive    image_goal_mission's SCAN / EXPLORE / APPROACH: scans, frontier exploration (within
           explore_radius_m of the start), a route through the local planner to a standoff
           short of the flag, the target masked out of the free-space profile.
  arrive   the live view: the flag seen within live_s, at most arrive_m away (organisers: 1 m).
  SDK      accepted -> the flag is marked (an obstacle, never a candidate again), next.
           Rejected once -> closer (the range scale may be wrong); rejected again at that flag
           -> not this checkpoint's flag: the next candidate.
  safety   a tilt, or wheels turning while the flag does not get closer: back off, mark ahead,
           reroute.

update() -> Step(carrot, command, arrived, state, note), as image_goal_mission.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from erc_inference import image_goal_mission as igm
from erc_inference.image_goal_mission import APPROACH, ARRIVED, SCAN, Goal, Step, _wrap

BACKOFF = 'backoff'
SHOW = 'show'

# the organisers' mechanics (29-sept): a flag scores when the rover SEES it from 3 m or less, judged
# by eye on the video, in any order (3 + 3 + 4 points); no SDK checkpoints. The node's defaults.
# Also from the indoor run (29-sept), where false cones 4-12 m away (behind walls, outside) filled the map and a
# target behind a step was chased for minutes: only sightings within 6 m (the arena is 8 x 8 m), 4 of them to
# believe a flag, and 90 s per flag.
FLAG_COMPETITION = {'any_order': True, 'show_s': 5.0, 'arrive_m': 1.5, 'max_shows': 6,
                    'max_detection_m': 6.0, 'min_confirm': 4, 'target_timeout_s': 90.0}

FLAG_DEFAULTS = dict(
    any_order=False,            # the nearest unvisited flag, no photo ranking, no SDK sorting
    max_detection_m=10.0,       # sightings further than this are ignored
    target_timeout_s=0.0,       # leave a flag after this long as the target without showing it (0: never) ...
    skip_s=180.0,               # ... for this long
    show_s=0.0,                 # at arrive_m: face it and hold still this long for the judges
    show_face_deg=30.0,
    max_shows=0,                # any order: keep showing new flags up to this many (not only 3: a show
                                # can miss -- a flag counted twice, one seen from over 3 m -- and an extra
                                # one costs only time)
    shown_apart_m=1.5,          # any order: a flag this close to where one was shown IS that one (the map
                                # splits a flag in two when the wheels slip; showing it twice scores nothing)
    flag_height_m=0.15,         # the cloth's vertical extent the detector assumed (its ranges use it)
    gate_deg=4.0,               # a sighting is that flag if its bearing is within max(this, 0.4 m at
    gate_lateral_m=0.6,         #   the flag's distance) of the flag's, and ...
    gate_range_ratio=2.0,       # ... its range within this factor (the range scale may be wrong)
    merge_m=0.8,                # two flags whose estimates come this close are one
    fit_min_samples=6,          # size fit: this many sightings ...
    fit_min_ratio=1.4,          # ... spanning this ratio of apparent sizes (it drove towards it)
    fit_prior_w=0.5,            # weight of the assumed height in the fit
    size_min_m=0.06,
    size_max_m=0.40,
    live_s=3.0,                 # a sighting this recent (capture time) is the live view: the picture itself
                                # lags 0.5-1.5 s, and the rest-point prediction accounts for its age
    stop_lag_s=1.5,             # the rover keeps going this long after a stop (control delay 1.3-1.5 s,
                                # 24-sept): arrival is called when the PREDICTED rest point is arrive_m away
    blind_arrive_frac=0.0,      # arrive by the map estimate alone this close (0: never; the map drifts
                                # when the wheels slip on rocks, the live view does not)
    regain_s=3.0,               # this close to it and not in view: face it; nothing for this long -> rescan
    regain_m=1.2,
    live_near_m=2.5,            # in view this close: the target is where the latest sighting puts it
    mask_m=0.3,                 # free space: only this much around the target is masked (not a wedge:
                                # the rock in front of a flag is real)
    mark_min_points=4,          # a flag seen this often is an obstacle on the map (never driven over)
    mark_move_m=0.25,           # ...re-marked when its estimate moves this much
    closer_target_m=0.45,       # a first rejection with the range scale still assumed: a straight pulse to this
    closer_v=0.25,
    on_flag_m=0.35,             # a known flag this close and not in view: the rover is on it -> back off
    photo_min_inliers=15,       # a frame matches a goal photo
    photo_fov_deg=50.0,         # a flag within this of the heading was in that frame ...
    photo_bearing_deg=20.0,     # ... and within this of where the photo's scene is in it
    photo_range_m=8.0,
    explore_radius_m=6.0,
    stuck_window_s=8.0,         # approaching: odometry moved stuck_odom_m but the flag got
    stuck_odom_m=0.8,           # less than stuck_closer_m closer -> wheels spinning
    stuck_closer_m=0.15,
    backoff_s=1.8,
    backoff_v=-0.25,
    mark_from_m=0.25,
    mark_to_m=0.7,
    mark_half_m=0.25,
)


@dataclass(eq=False)          # a flag is itself, not its contents
class Flag:
    points: List[np.ndarray] = field(default_factory=list)
    samples: List[tuple] = field(default_factory=list)     # (x, y, world bearing, tan-span)
    visited: bool = False
    rejected: Dict[int, int] = field(default_factory=dict)  # goal index -> rejections
    wrong: Set[int] = field(default_factory=set)            # goal indices it is not
    score: Dict[int, int] = field(default_factory=dict)     # goal index -> best photo inliers
    live_t: float = -math.inf
    live_range: float = math.inf
    height: float = float('nan')                            # fitted cloth height
    fit_xy: Optional[np.ndarray] = None
    marked: Optional[tuple] = None                          # the map cell it is marked in
    skip_until: float = -math.inf                           # left (not reached in time) until then
    live_pt: Optional[np.ndarray] = None                    # where the latest sighting puts it

    def estimate(self, n=8):
        if self.fit_xy is not None:
            return self.fit_xy
        return np.median(np.array(self.points[-n:]), axis=0)


class FlagMission(igm.ImageGoalMission):
    def __init__(self, n_checkpoints: int = 3, start_xy=(0.0, 0.0), lp_params=None, recovery_params=None,
                 names: Optional[Sequence[str]] = None, **kw):
        if kw.get('any_order'):
            n_checkpoints = max(n_checkpoints, int(kw.get('max_shows', 0)))
        goals = [Goal(names[i] if names and i < len(names) else f'CP{i + 1}', 'blue', i + 1)
                 for i in range(n_checkpoints)]
        base = {k: v for k, v in kw.items() if k in igm.DEFAULTS}
        base.setdefault('scan_step_deg', 90.0)
        base.setdefault('map_half_m', 15.0)
        base.setdefault('explore_hop_m', 2.0)
        base.setdefault('frontier_max_m', 4.0)
        base.setdefault('scan_every_m', 3.0)
        base.setdefault('standoff_m', 0.55)
        base.setdefault('arrive_m', 0.5)          # where it should come to rest (organisers: within 1 m)
        base.setdefault('reroute_moved_m', 0.3)
        base.setdefault('carrot_distance_m', 1.0)
        unknown = set(kw) - set(igm.DEFAULTS) - set(FLAG_DEFAULTS)
        if unknown:
            raise ValueError(f'unknown flag mission parameters: {sorted(unknown)}')
        self.flags: List[Flag] = []
        self.target: Optional[Flag] = None
        self._t = -math.inf
        self._backoff_until = None
        self._pulse_until = None
        self._show_phase, self._show_t0 = None, 0.0
        self._shown = []
        self._timed, self._timed_t0 = None, 0.0
        self._on_flag_t = -math.inf
        self._poses = []
        self._approach_trail = []
        self.stucks = 0
        super().__init__(goals, start_xy, lp_params, recovery_params, **base)
        self.p.update(FLAG_DEFAULTS)
        self.p.update({k: v for k, v in kw.items() if k in FLAG_DEFAULTS})
        self._arrive0 = self.p['arrive_m']
        self._standoff0 = self.p['standoff_m']

    # -- the flag map ---------------------------------------------------------

    def _begin_leg(self, t):
        super()._begin_leg(t)
        self.target = None
        self._approach_trail = []
        self._regain_t0 = None

    def observe_scan(self, t, x, y, yaw, bearings_deg, ranges, hit) -> bool:
        b = np.asarray(bearings_deg, float)
        r = np.asarray(ranges, float)
        h = np.asarray(hit, bool).copy()
        est = self.estimate()
        if est is not None:
            d = float(np.hypot(*(est - (x, y))))
            brg = math.degrees(_wrap(math.atan2(est[1] - y, est[0] - x) - yaw))
            half = math.degrees(math.atan2(self.p['mask_m'], max(d, 0.3)))
            h[(np.abs(b - brg) <= half) & (np.abs(r - d) <= self.p['mask_m'])] = False
        return self.replanner.observe(t, x, y, yaw, b, r, h)

    def estimate(self):
        return self.est(self.target) if self.target is not None else None

    def observe_cone(self, x, y, yaw, cone_class, bearing_deg, range_m, t=0.0, ang_height=None) -> bool:
        """One flag sighting from the estimated pose (the cone interface: /erc/flags has its shape)."""
        if cone_class != 'blue' or not math.isfinite(range_m) or range_m > self.p['max_detection_m']:
            return False
        p = self.p
        self._t = max(self._t, t)
        range_m = range_m * self.height_scale()          # all three flags are the same model
        a = yaw + math.radians(bearing_deg)
        pt = np.array([x + range_m * math.cos(a), y + range_m * math.sin(a)])
        # by bearing first: it is exact to ~1 deg, the range only to its (assumed) scale
        best, best_db = None, math.inf
        for f in self.flags:
            e = self.est(f)
            d = float(np.hypot(e[0] - x, e[1] - y))
            if d < 1e-3:
                continue
            db = abs(math.degrees(_wrap(math.atan2(e[1] - y, e[0] - x) - a)))
            gate = max(p['gate_deg'], math.degrees(math.atan2(p['gate_lateral_m'], d)))
            ratio = max(range_m, 1e-3) / d
            if db <= gate and 1.0 / p['gate_range_ratio'] <= ratio <= p['gate_range_ratio'] and db < best_db:
                best, best_db = f, db
        if best is None:
            best = Flag()
            self.flags.append(best)
        best.points.append(pt)
        del best.points[:-30]
        if ang_height is not None and ang_height > 0:
            best.samples.append((x, y, a, float(ang_height)))
            del best.samples[:-60]
            self._fit(best)
        best.live_t = t
        best.live_range = self._range(best, range_m, ang_height)
        best.live_pt = np.array([x + best.live_range * math.cos(a), y + best.live_range * math.sin(a)])
        self._merge(best)
        self._mark_flag(best)
        return True

    def _merge(self, f: Flag):
        for g in list(self.flags):
            if g is f or g.visited or f.visited:
                continue
            if float(np.hypot(*(g.estimate() - f.estimate()))) <= self.p['merge_m']:
                f.points = (g.points + f.points)[-30:]
                f.samples = (g.samples + f.samples)[-60:]
                f.wrong |= g.wrong
                for k, v in g.rejected.items():
                    f.rejected[k] = max(f.rejected.get(k, 0), v)
                for k, v in g.score.items():
                    f.score[k] = max(f.score.get(k, 0), v)
                if g.marked is not None:
                    m = self.replanner.map
                    r, c = m.cell([g.marked[0]], [g.marked[1]])
                    if m._inside(r, c).all():
                        m.L[r, c] = 0.0
                        m._dist = None
                if self.target is g:
                    self.target = f
                self.flags.remove(g)
                self._fit(f)

    def _mark_flag(self, f: Flag):
        """Every flag is an obstacle: a thin wire the rover must not drive over (it would knock it
        down). Moved when its estimate does."""
        if len(f.points) < self.p['mark_min_points']:
            return
        e = f.estimate()
        if f.marked is not None and math.hypot(e[0] - f.marked[0], e[1] - f.marked[1]) < self.p['mark_move_m']:
            return
        m = self.replanner.map
        if f.marked is not None:
            r, c = m.cell([f.marked[0]], [f.marked[1]])
            if m._inside(r, c).all():
                m.L[r, c] = 0.0
                m._dist = None
        r, c = m.cell([e[0]], [e[1]])
        if m._inside(r, c).all():
            m.L[r, c] = m.p['max_logodds']
            m.seen[r, c] = True
            m._dist = None
        f.marked = (float(e[0]), float(e[1]))

    def _range(self, f: Flag, range_m, ang_height):
        if ang_height and math.isfinite(f.height):
            return f.height / ang_height
        return range_m

    def height_scale(self):
        """Fitted cloth height over the assumed one, from every flag with a fit (same model)."""
        h = [f.height for f in self.flags if math.isfinite(f.height)]
        return float(np.median(h)) / self.p['flag_height_m'] if h else 1.0

    def est(self, f: Flag):
        """Where flag f is: the latest sighting when it is in view and near (the map drifts with
        wheel slip, the live view does not), else the fit, else the recent sightings."""
        if f.live_pt is not None and self._t - f.live_t <= self.p['live_s'] and f.live_range <= self.p['live_near_m']:
            return f.live_pt
        return f.estimate()

    def _fit(self, f: Flag):
        """Position and cloth height together: pos_i + (H / a_i) u_i = P for every sighting."""
        p = self.p
        S = np.array(f.samples)
        if len(S) < p['fit_min_samples'] or S[:, 3].max() / S[:, 3].min() < p['fit_min_ratio']:
            return
        n = len(S)
        A = np.zeros((2 * n + 1, 3))
        b = np.zeros(2 * n + 1)
        A[0:2 * n:2, 0] = 1.0
        A[1:2 * n:2, 1] = 1.0
        A[0:2 * n:2, 2] = -np.cos(S[:, 2]) / S[:, 3]
        A[1:2 * n:2, 2] = -np.sin(S[:, 2]) / S[:, 3]
        b[0:2 * n:2] = S[:, 0]
        b[1:2 * n:2] = S[:, 1]
        # the prior, in metres of position error per metre of height: ~1/a of a typical sighting
        w = p['fit_prior_w'] / float(np.median(S[:, 3]))
        A[-1, 2], b[-1] = w, w * p['flag_height_m']
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        if p['size_min_m'] <= sol[2] <= p['size_max_m']:
            f.fit_xy, f.height = sol[:2], float(sol[2])

    def observe_scene(self, t, x, y, yaw, scores):
        """Photo matches for a frame taken from (x, y, yaw), one per goal: (inliers, bearing of the
        matched scene in the frame, deg) or just inliers. Credited to the flags in that frame's
        view, near the matched scene when its bearing is known."""
        for f in self.flags:
            e = self.est(f)
            d = float(np.hypot(e[0] - x, e[1] - y))
            if d > self.p['photo_range_m']:
                continue
            fb = math.degrees(_wrap(math.atan2(e[1] - y, e[0] - x) - yaw))
            if abs(fb) > self.p['photo_fov_deg']:
                continue
            for k, sc in enumerate(scores):
                n, sb = (sc if isinstance(sc, (tuple, list)) else (sc, float('nan')))
                if n < self.p['photo_min_inliers']:
                    continue
                if sb == sb and abs(fb - sb) > self.p['photo_bearing_deg']:
                    continue
                f.score[k] = max(f.score.get(k, 0), int(n))

    def _candidates(self, x, y):
        if self.p['any_order']:
            for f in self.flags:
                if not f.visited and any(float(np.hypot(*(self.est(f) - q))) <= self.p['shown_apart_m']
                                         for q in self._shown):
                    f.visited = True             # the one already shown, mapped twice
            c = [f for f in self.flags if not f.visited and len(f.points) >= self.p['min_confirm']
                 and f.skip_until <= self._t]
            c.sort(key=lambda f: float(np.hypot(*(self.est(f) - (x, y)))))
            return c
        k = self.index
        c = [f for f in self.flags if not f.visited and k not in f.wrong and len(f.points) >= self.p['min_confirm']]
        c.sort(key=lambda f: (-f.score.get(k, 0), float(np.hypot(*(self.est(f) - (x, y))))))
        return c

    def _pick_target(self, t, x, y):
        c = self._candidates(x, y)
        if not c:
            return
        best = c[0]
        if self.target is None or self.target not in c:
            self.target = best
        elif best is not self.target and best.score.get(self.index, 0) > self.target.score.get(self.index, 0) + 10:
            self.target = best              # the photo points at another flag: switch
        else:
            return
        self.p['arrive_m'], self.p['standoff_m'] = self._arrive0, self._standoff0   # a new flag: afresh
        e = self.est(self.target)
        why = f'photo {self.target.score[self.index]} inliers' if self.target.score.get(self.index) else 'nearest'
        self.notes.append(f'{t:.1f} {self.goal.name}: flag at ({e[0]:.1f},{e[1]:.1f}) ({why}; '
                          f'{len(self.flags)} flags on the map)')
        self._route_to = None
        self._approach_trail = []

    # -- SDK ------------------------------------------------------------------

    def confirm(self, t, accepted: bool):
        g, f = self.goal, self.target
        if accepted:
            if f is not None:
                f.visited = True
            self.notes.append(f'{t:.1f} {g.name if g else "?"} confirmed')
            self.index += 1
            self.p['arrive_m'] = self._arrive0
            self._begin_leg(t)
            return
        if f is None:
            self.state = SCAN
            return
        f.rejected[self.index] = f.rejected.get(self.index, 0) + 1
        calibrated = math.isfinite(f.height) or self.height_scale() != 1.0
        if f.rejected[self.index] == 1 and not calibrated and f.live_range > self.p['closer_target_m'] + 0.1 \
                and t - f.live_t <= 2 * self.p['live_s']:
            # the range scale is still the assumed one: it may really be over 1 m. A timed straight
            # pulse (the route overshoots onto the flag: ~1 s of image lag), then ask again.
            dur = (f.live_range - self.p['closer_target_m']) / self.p['closer_v']
            self._pulse_until = t + dur
            self.state = ARRIVED
            self._arrived_note = f'{g.name}: closer by {f.live_range - self.p["closer_target_m"]:.2f} m'
            self.notes.append(f'{t:.1f} rejected with the range scale unknown; {self._arrived_note}')
            return
        f.wrong.add(self.index)
        self.target = None
        self.p['arrive_m'], self.p['standoff_m'] = self._arrive0, self._standoff0
        self.state = SCAN
        self._full_scan_pending = not self._candidates(*self.est(f))
        self.notes.append(f'{t:.1f} rejected: not {g.name}\'s flag; next candidate')

    @property
    def igm_standoff(self):
        return self._standoff0

    # -- safety -----------------------------------------------------------------

    def _mark_ahead(self, x, y, yaw):
        p = self.p
        m = self.replanner.map
        ds = np.arange(p['mark_from_m'], p['mark_to_m'] + 1e-9, m.p['resolution_m'] * 0.5)
        dl = np.arange(-p['mark_half_m'], p['mark_half_m'] + 1e-9, m.p['resolution_m'] * 0.5)
        S, Lg = np.meshgrid(ds, dl)
        xs = x + S * math.cos(yaw) - Lg * math.sin(yaw)
        ys = y + S * math.sin(yaw) + Lg * math.cos(yaw)
        rows, cols = m.cell(xs.ravel(), ys.ravel())
        ok = m._inside(rows, cols)
        m.L[rows[ok], cols[ok]] = m.p['max_logodds']
        m.seen[rows[ok], cols[ok]] = True
        m._dist = None

    def block_ahead(self, t, x, y, yaw, why):
        """A tilt, or wheels spinning: back off, mark ahead, reroute."""
        self.stucks += 1
        self._mark_ahead(x, y, yaw)
        self._backoff_until = t + self.p['backoff_s']
        self._route_to = None
        self._approach_trail = []
        self.notes.append(f'{t:.1f} {why}: backing off, marked ahead')

    def _check_stuck(self, t, x, y, yaw):
        f = self.target
        if self.state != APPROACH or f is None or t - f.live_t > self.p['live_s']:
            self._approach_trail = []
            return False
        self._approach_trail = [e for e in self._approach_trail if t - e[0] <= self.p['stuck_window_s']]
        self._approach_trail.append((t, x, y, f.live_range))
        t0, x0, y0, r0 = self._approach_trail[0]
        if t - t0 < self.p['stuck_window_s'] - 0.5:
            return False
        if math.hypot(x - x0, y - y0) >= self.p['stuck_odom_m'] and r0 - f.live_range < self.p['stuck_closer_m']:
            self.block_ahead(t, x, y, yaw, f'stuck: odometry {math.hypot(x - x0, y - y0):.1f} m, '
                                           f'flag {r0 - f.live_range:+.2f} m closer')
            return True
        return False

    # -- the step -----------------------------------------------------------------

    def update(self, t, x, y, yaw, rate=0.0, reflex_idle=True) -> Step:
        p = self.p
        if self.done:
            return Step(None, (0.0, 0.0), False, 'done', 'mission complete')
        if self._backoff_until is not None:
            if t < self._backoff_until:
                return Step(None, (p['backoff_v'], 0.0), False, BACKOFF, 'backing off')
            self._backoff_until = None
        if self._pulse_until is not None:
            if t < self._pulse_until:
                return Step(None, (p['closer_v'], 0.0), False, APPROACH, 'closer')
            self._pulse_until = None
            self.notes.append(f'{t:.1f} arrived {self._arrived_note}')
            return Step(None, (0.0, 0.0), True, ARRIVED, self._arrived_note)
        if self.state == ARRIVED:
            return Step(None, (0.0, 0.0), True, ARRIVED, self._arrived_note)
        self._t = max(self._t, t)
        if t - self._on_flag_t > 10.0:
            for q in self.flags:
                if t - q.live_t > p['live_s'] and float(np.hypot(*(q.estimate() - (x, y)))) < p['on_flag_m']:
                    self._on_flag_t = t
                    self._backoff_until = t + p['backoff_s']
                    self.notes.append(f'{t:.1f} a flag is under the rover (not in view): backing off')
                    return Step(None, (p['backoff_v'], 0.0), False, BACKOFF, 'off the flag')
        self._poses = [e for e in self._poses if t - e[0] <= 1.0] + [(t, x, y)]
        v = 0.0
        if self._poses[-1][0] - self._poses[0][0] > 0.4:
            v = math.hypot(x - self._poses[0][1], y - self._poses[0][2]) / (self._poses[-1][0] - self._poses[0][0])
        self._pick_target(t, x, y)
        if self.target is not self._timed:
            self._timed, self._timed_t0 = self.target, t
        f = self.target
        if (f is not None and p['target_timeout_s'] > 0 and t - self._timed_t0 > p['target_timeout_s']
                and self.state not in (SHOW, ARRIVED)):
            # unreachable from here (behind a rock, up a slope it cannot take) or not a flag at all: leave it
            # for skip_s -- another flag first; it may be reachable from elsewhere later
            e = self.est(f)
            f.skip_until = t + p['skip_s']
            self.notes.append(f'{t:.1f} flag at ({e[0]:.1f},{e[1]:.1f}): not reached in {p["target_timeout_s"]:.0f} s; '
                              f'leaving it for {p["skip_s"]:.0f} s')
            self.target, self._timed = None, None
            self.p['arrive_m'], self.p['standoff_m'] = self._arrive0, self._standoff0
            self.state = SCAN
            self._full_scan_pending = False
            self._scan_queue, self._scan_target = [], None
            return Step(None, (0.0, 0.0), False, SCAN, 'target given up')
        rest = f.live_range - v * (max(0.0, t - f.live_t) + p['stop_lag_s']) if f is not None else math.inf
        if self.state == SHOW:
            return self._show(t, x, y, yaw, rate)
        if f is not None and t - f.live_t <= p['live_s'] and rest <= p['arrive_m'] and p['show_s'] > 0:
            self.state = SHOW
            self._show_phase, self._show_t0 = 'face', t
            self.notes.append(f'{t:.1f} flag {f.live_range:.2f} m away, at rest ~{max(rest, 0):.2f} m; showing it')
            return self._show(t, x, y, yaw, rate)
        if f is not None and t - f.live_t <= p['live_s'] and rest <= p['arrive_m']:
            self.state = ARRIVED
            h = f' (cloth {f.height:.2f} m fitted)' if math.isfinite(f.height) else ''
            self._arrived_note = f'{self.goal.name}: flag {f.live_range:.2f} m away, at rest ~{max(rest, 0):.2f} m{h}'
            self.notes.append(f'{t:.1f} arrived {self._arrived_note}')
            return Step(None, (0.0, 0.0), True, ARRIVED, self._arrived_note)
        if self._check_stuck(t, x, y, yaw):
            return Step(None, (p['backoff_v'], 0.0), False, BACKOFF, 'stuck')
        regain = self._regain(t, x, y, yaw)
        if regain is not None:
            return regain
        # the base class arrives by the map estimate; the map is only as good as the range scale,
        # so without the live view that is allowed only closer
        live_arrive = p['arrive_m']
        p['arrive_m'] = live_arrive * p['blind_arrive_frac']
        try:
            step = super().update(t, x, y, yaw, rate, reflex_idle)
        finally:
            p['arrive_m'] = live_arrive
        return step

    def _regain(self, t, x, y, yaw):
        """At the route's end without the flag in view: face where it should be; still nothing
        after regain_s -> that estimate is stale (odometry slipped): drop it and scan."""
        p = self.p
        f = self.target
        if self.state != APPROACH or f is None or t - f.live_t <= p['live_s']:
            self._regain_t0 = None
            return None
        e = self.est(f)
        d = float(np.hypot(e[0] - x, e[1] - y))
        if d > p['regain_m']:
            self._regain_t0 = None
            return None
        if self._regain_t0 is None:
            self._regain_t0 = t
        err = _wrap(math.atan2(e[1] - y, e[0] - x) - yaw)
        if abs(err) > math.radians(20.0) and t - self._regain_t0 < 2 * p['regain_s']:
            return Step(None, (0.0, math.copysign(p['scan_w'], err)), False, APPROACH, 'facing the flag')
        if t - self._regain_t0 < p['regain_s']:
            return Step(None, (0.0, 0.0), False, APPROACH, 'looking for the flag')
        self.notes.append(f'{t:.1f} {self.goal.name}: flag not where the map says; dropped it, scanning')
        if not f.visited:
            if f.marked is not None:
                m = self.replanner.map
                r, c = m.cell([f.marked[0]], [f.marked[1]])
                if m._inside(r, c).all():
                    m.L[r, c] = 0.0
                    m._dist = None
            self.flags = [q for q in self.flags if q is not f]
        self.target = None
        self._regain_t0 = None
        self.p['arrive_m'], self.p['standoff_m'] = self._arrive0, self._standoff0
        self.state = SCAN
        self._full_scan_pending = False
        self._scan_queue, self._scan_target = [], None
        return Step(None, (0.0, 0.0), False, SCAN, 'rescan')

    def _show(self, t, x, y, yaw, rate):
        """Face the flag (well inside the image), then hold still show_s for the judges."""
        p = self.p
        f = self.target
        if self._show_phase == 'face' and f is not None:
            e = self.est(f)
            err = _wrap(math.atan2(e[1] - y, e[0] - x) - yaw)
            lead = abs(rate or 0.0) * p['scan_lead_s']
            if abs(err) > max(math.radians(p['show_face_deg']), lead) and t - self._show_t0 < 10.0:
                return Step(None, (0.0, math.copysign(p['scan_w'], err)), False, SHOW, 'facing the flag')
        if self._show_phase == 'face':
            self._show_phase, self._show_t0 = 'hold', t
        if t - self._show_t0 < p['show_s']:
            return Step(None, (0.0, 0.0), False, SHOW, 'showing the flag')
        self.state = ARRIVED
        if f is not None:
            self._shown.append(self.est(f).copy())
        self._arrived_note = f'flag {self.index + 1} shown'
        self.notes.append(f'{t:.1f} {self._arrived_note}')
        return Step(None, (0.0, 0.0), True, ARRIVED, self._arrived_note)

    def _plan_scan(self, yaw, full):
        """There and back rather than a full circle: the turns cancel, and with them the gyro's
        scale error (a full circle at 3% is 11 deg of heading; loop_patrol_sim lost the pose to it)."""
        if not full:
            return super()._plan_scan(yaw, full)
        h = math.pi / 2
        self._scan_queue = [yaw + h, yaw + 2 * h, yaw + h, yaw, yaw - h, yaw]
        self._scan_target = None
        self._scan_phase = None

    # -- explore, bounded (as homing_mission) ------------------------------------

    def _pick_hop(self, t, x, y, yaw, choice=None):
        r = float(np.hypot(x - self.start[0], y - self.start[1]))
        if r > self.p['explore_radius_m']:
            a = math.atan2(self.start[1] - y, self.start[0] - x)
            L = min(self.p['explore_hop_m'], r)
            self._set_route([(x, y), (x + L * math.cos(a), y + L * math.sin(a))])
            self._hop_t0, self._repick = t, False
            self.notes.append(f'{t:.1f} explore: {self.p["explore_radius_m"]:.0f} m from the start; back towards it')
            return True
        ok = super()._pick_hop(t, x, y, yaw, choice)
        if ok and self.pts is not None:
            d = np.hypot(self.pts[:, 0] - self.start[0], self.pts[:, 1] - self.start[1])
            if d[-1] > self.p['explore_radius_m']:
                keep = self.pts[d <= self.p['explore_radius_m']]
                self._set_route(keep if len(keep) >= 2 else [(x, y), (x, y)])
        return ok
