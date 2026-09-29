"""Off-road image-goal missions: drive to where each goal photo was taken. No map, no GPS.

ERC 2026 off-road track: each mission is a target image; the rover must get there from the
camera alone, over slopes, gravel and rocks (the Verti-Arena kind of terrain: 8x8 m, 0.7 m
of relief, boulders that roll a rover). Time is only the tiebreaker; tipping over is the one
outcome that cannot happen. No ROS: image_goal_offroad_node and test/homing_sim.py run this.

What the camera gives (goal_homing.GoalMatcher, one frame at a time): how many features
match the goal image, the bearing to where it was taken, the goal's heading relative to the
rover's, and the parallax -- what is left of the image motion once the heading difference is
taken out, ~0 at the goal. The bearing is only good when the parallax is large enough; the
distance is unknown without metric depth. So the mission is stop-and-go:

  SCAN     turn in place in scan_step_deg steps, standing at each; keep the stop that saw the
           goal's scene best. Seen -> MEASURE there. Not seen -> EXPLORE.
  MEASURE  stand still settle_s (the picture lags the pose by up to ~1 s), then take the
           median of measure_n matches. Parallax under arrive_parallax_px -> ALIGN, then
           ARRIVED. Otherwise HOP. Nothing matches -> partial scan, then full scan.
  HOP      a hop along the measured bearing: hop_per_px m per pixel of parallax,
           hop_min_m..hop_max_m (or the PnP distance when there is depth). Driven as a route
           through the local planner (detours around what /erc/free_space maps), with the
           carrot.
  PULSE    near the goal (parallax under near_px): turn to the bearing, then a timed straight
           pulse at the speed floor (a carrot cannot place the rover to 0.2 m with 1.3 s of
           lag), only if the map is clear that way. Close behind: straight back.
  ALIGN    turn to the goal photo's heading before measuring, so the scene is in view.
  EXPLORE  frontier hops (image_goal_mission), biased towards the scan stop with the most raw
           matches, never further than explore_radius_m from the start; scan after each.

No progress (the parallax did not shrink over no_progress_stops stops, e.g. wheels spinning
on a slope that odometry believes it climbed) marks the way ahead as blocked and detours to
one side. A tilt (safety envelope) backs off and marks it the same way.

update() -> Step(carrot, command, arrived, state, note), as image_goal_mission.
"""
import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from erc_inference import image_goal_mission as igm
from erc_inference.image_goal_mission import Step, _wrap

SCAN, MEASURE, HOP, PULSE, ALIGN, EXPLORE, ARRIVED, BACKOFF = (
    'scan', 'measure', 'hop', 'pulse', 'align', 'explore', 'arrived', 'backoff')
MEASURE_AFTER_HOP = 'measure-after-hop'     # ALIGN's follow-up: a stop that checks it moved

HOMING_DEFAULTS = dict(
    min_inliers=20,             # a match with this many inliers sees the goal's scene
    sighting_inliers=25,        # ...and this many while exploring or hopping: stop and measure there
    arrive_parallax_px=12.0,    # standing where the photo was taken: under ~0.3 m in homing_sim's arena
                                # (it depends on how far the scene is: 6 m away, 0.66 m reads 16 px)
    arrive_min_inliers=30,
    arrive_max_log_zoom=0.03,   # and the SIFT scales within 3% of the goal's (when there are scales)
    zoom_step_m=0.3,            # parallax says here, the scales say behind / ahead: this far along the photo's heading
    settle_s=1.6,               # stand still this long before a measurement counts
    scan_tail_s=1.6,            # after the last scan stop, wait this long for its (late) frames
    measure_n=3,
    measure_timeout_s=5.0,      # at a stop, waiting for measure_n matches
    hop_min_m=0.5,              # a routed hop (planner + carrot)
    hop_max_m=1.5,
    hop_per_px=0.01,            # m of hop per px of parallax (homing_sim: ~60-110 px/m at 0.5-3 m)
    near_px=60.0,               # under this parallax: timed straight pulses instead of routed hops
    near_gain=0.8,              # a pulse covers this share of the parallax-estimated distance
    pulse_min_m=0.12,
    pulse_max_m=0.6,
    pulse_v=0.25,               # the rover's floor
    pulse_loss_m=0.05,          # distance the shaper's ramps take off a pulse (0.3 / 0.6 m/s^2)
    reverse_max_m=0.8,          # a goal behind, this close: back up instead of turning round
    no_progress_stops=3,        # stops without the parallax falling below progress_frac of the best
    progress_frac=0.85,
    progress_min_px=45.0,       # under this the parallax is mostly noise: no progress test
    detour_deg=60.0,
    align_heading=True,
    align_tol_deg=8.0,          # the final alignment to the photo's heading
    view_tol_deg=35.0,          # between hops: the scene stays in view (camera +-60 deg) within this
    explore_radius_m=6.0,       # frontiers further than this from the start are not explored
    # stuck: after a hop, this stop's frame against the previous stop's (the node matches them):
    # a derotated parallax under still_px means the rover did not move, whatever the wheels say
    still_px=6.0,
    still_min_odom_m=0.25,      # ...when odometry says it moved at least this much
    motion_timeout_s=3.0,
    # tilt / stuck: back off, mark ahead, hop elsewhere
    backoff_s=1.8,
    backoff_v=-0.25,
    mark_from_m=0.25,
    mark_to_m=0.7,
    mark_half_m=0.25,
)


@dataclass
class HomingGoal:
    name: str
    image: str = ''
    sequence: int = 0
    is_start_cone: bool = False     # image_goal_mission's frontier picking asks


@dataclass
class Measurement:
    t: float
    x: float
    y: float
    yaw: float
    h: object                       # goal_homing.Homing


class HomingMission(igm.ImageGoalMission):
    def __init__(self, goals: Sequence[HomingGoal], start_xy=(0.0, 0.0), lp_params=None, recovery_params=None,
                 **kw):
        base = {k: v for k, v in kw.items() if k in igm.DEFAULTS}
        base.setdefault('scan_step_deg', 90.0)
        base.setdefault('map_half_m', 15.0)
        base.setdefault('explore_hop_m', 2.0)
        base.setdefault('frontier_max_m', 4.0)
        base.setdefault('scan_every_m', 3.0)
        super().__init__(goals, start_xy, lp_params, recovery_params, **base)
        self.p.update(HOMING_DEFAULTS)
        self.p.update({k: v for k, v in kw.items() if k in HOMING_DEFAULTS})
        unknown = set(kw) - set(igm.DEFAULTS) - set(HOMING_DEFAULTS)
        if unknown:
            raise ValueError(f'unknown homing parameters: {sorted(unknown)}')
        self._arrive_px0 = self.p['arrive_parallax_px']
        self.stops = 0
        self.stucks = 0

    # -- leg bookkeeping ----------------------------------------------------

    def _begin_leg(self, t):
        super()._begin_leg(t)
        self.meas: List[Measurement] = []
        self._stop_t0 = None
        self._stop_meas: List[Measurement] = []
        self._scan_seen: List[Measurement] = []     # measurements taken standing at a scan stop
        self._scan_windows = {}                     # settle start -> last time seen settling
        self._scan_done_t = None
        self._best_parallax = math.inf
        self._no_progress = 0
        self._detour_side = 1
        self._after_align = MEASURE
        self._pulse = None                          # (t_end, v) of a timed straight pulse
        self._pulse_plan = None                     # (duration, v) to run once aligned
        self._backoff_until = None
        self._lost_scans = 0
        self._hint = None                           # explore bias: a direction (rad, world)
        self._sighted = None                        # a confident match while exploring
        self._stop_pose = None                      # odometry at the last stop
        self._hop_from = None                       # (x, y, yaw, heading of the hop) it left from
        self._motion_px = None                      # this stop vs the last one (observe_motion)
        self._motion_wanted = False
        self._last = None                           # the last good stop's median measurement
        self._arrived_note = ''

    def confirm(self, t, accepted: bool):
        """The SDK's answer after ARRIVED. Accepted: next goal. Rejected: a tighter parallax."""
        g = self.goal
        if accepted:
            self.notes.append(f'{t:.1f} {g.name if g else "?"} confirmed')
            self.index += 1
            self.p['arrive_parallax_px'] = self._arrive_px0
            self._begin_leg(t)
        else:
            self.p['arrive_parallax_px'] = max(3.0, 0.6 * self.p['arrive_parallax_px'])
            self._start_stop(t)
            self.notes.append(f'{t:.1f} rejected; arriving now needs parallax under '
                              f'{self.p["arrive_parallax_px"]:.0f} px')

    def estimate(self):
        return None                 # no cone: image_goal_mission's cone logic stays idle

    # -- inputs -------------------------------------------------------------

    def observe_scan(self, t, x, y, yaw, bearings_deg, ranges, hit) -> bool:
        return self.replanner.observe(t, x, y, yaw, np.asarray(bearings_deg, float), np.asarray(ranges, float),
                                      np.asarray(hit, bool))

    @property
    def wants_motion_check(self):
        """The node should match a frame of this stop against the previous stop's frame."""
        return self._motion_wanted and self._motion_px is None

    def observe_motion(self, t, parallax_px) -> None:
        """Derotated parallax (px) between the previous stop's frame and one of this stop (NaN:
        they do not match at all -- it moved)."""
        if self._motion_wanted and self._motion_px is None:
            self._motion_px = float(parallax_px)

    def observe_homing(self, t, x, y, yaw, h) -> None:
        """One GoalMatcher answer, for a frame taken at time t from pose (x, y, yaw)."""
        m = Measurement(t, x, y, yaw, h)
        self.meas.append(m)
        del self.meas[:-200]
        standing = self._stop_t0 is not None and t >= self._stop_t0 + self.p['settle_s']
        if self.state == SCAN:
            self._scan_seen.append(m)       # sorted out by capture time when the scan ends
        elif self.state == MEASURE and standing:
            self._stop_meas.append(m)
        elif self.state == EXPLORE and h.ok and h.inliers >= self.p['sighting_inliers'] and self._sighted is None:
            self._sighted = m

    # -- helpers -------------------------------------------------------------

    def _start_stop(self, t, after_hop=False):
        self.state = MEASURE
        self._stop_t0 = t
        self._stop_meas = []
        self._motion_wanted = after_hop and self._hop_from is not None
        self._motion_px = None

    def _median(self, ms):
        good = [m for m in ms if m.h.ok and m.h.inliers >= self.p['min_inliers']]
        if not good:
            return None, len(ms)
        par = float(np.median([m.h.parallax_px for m in good]))
        inl = int(np.median([m.h.inliers for m in good]))
        # angles: the measurement nearest the median parallax, not an average of wrapped angles
        k = int(np.argmin([abs(m.h.parallax_px - par) for m in good]))
        return (good[k], par, inl), len(ms)

    def _scan_pick(self):
        # by the time the picture was taken, not by when it arrived: with ~1 s of image lag the
        # frames of a stop come in while the rover is already turning to the next one
        win = list(self._scan_windows.items())
        self._scan_seen = [m for m in self._scan_seen if any(t0 + 0.5 <= m.t <= t1 + 0.2 for t0, t1 in win)]
        good = [m for m in self._scan_seen if m.h.ok and m.h.inliers >= self.p['min_inliers']]
        if good:
            return max(good, key=lambda m: m.h.inliers)
        if self._scan_seen:
            weak = max(self._scan_seen, key=lambda m: m.h.matches)
            if weak.h.matches > 0:
                self._hint = weak.yaw + (weak.h.view_bearing if weak.h.view_bearing == weak.h.view_bearing else 0.0)
        return None

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
        """Something ahead stopped the rover (a tilt, a slope it cannot climb): back off, mark it,
        and the next hop goes round it."""
        self.stucks += 1
        self._mark_ahead(x, y, yaw)
        self._backoff_until = t + self.p['backoff_s']
        self.state = BACKOFF
        self.notes.append(f'{t:.1f} {why}: backing off, marked ahead')

    def _plan_hop(self, t, x, y, yaw, m, par):
        """From a stop's measurement: a routed hop along the bearing, a timed pulse near the goal,
        a detour when there is no progress, or straight back when the goal is close behind."""
        p = self.p
        h = m.h
        a = m.yaw + h.bearing                          # world direction to the goal, from the stop
        est = h.dist if h.dist == h.dist else p['hop_per_px'] * par
        detour = self._no_progress >= p['no_progress_stops']
        if detour:
            self._mark_ahead(x, y, a)
            a += self._detour_side * math.radians(p['detour_deg'])
            self._detour_side = -self._detour_side
            self._no_progress = 0
            self._best_parallax = math.inf
            self.notes.append(f'{t:.1f} no progress: marked ahead, detour {math.degrees(_wrap(a - yaw)):+.0f} deg')
        rel = _wrap(a - yaw)
        if par < p['near_px'] and not detour:
            L = float(np.clip(p['near_gain'] * est, p['pulse_min_m'], p['pulse_max_m']))
            self._hop_from = (x, y, yaw, a)
            if abs(rel) > math.radians(120.0) and L <= p['reverse_max_m']:
                self._start_pulse(t, L, -1.0)
                self.notes.append(f'{t:.1f} goal {L:.2f} m behind: backing up (parallax {par:.0f} px)')
                return
            if self._clear_ahead(x, y, a, L + 0.2):
                self._pulse_plan = ((L + p['pulse_loss_m']) / p['pulse_v'], p['pulse_v'])
                self.notes.append(f'{t:.1f} pulse {L:.2f} m at {math.degrees(rel):+.0f} deg '
                                  f'(parallax {par:.0f} px, {h.inliers} inliers)')
                if abs(rel) > math.radians(10.0):
                    self._align_to(t, yaw, a, PULSE)
                else:
                    self._start_pulse(t, L, 1.0)
                return
        L = float(np.clip(est, p['hop_min_m'], p['hop_max_m']))
        self._hop_from = (x, y, yaw, a)
        self._set_route([(x, y), (x + L * math.cos(a), y + L * math.sin(a))])
        self.state = HOP
        self.notes.append(f'{t:.1f} hop {L:.2f} m at {math.degrees(rel):+.0f} deg '
                          f'(parallax {par:.0f} px, {h.inliers} inliers)')

    def _start_pulse(self, t, L=None, sign=1.0):
        p = self.p
        if L is not None:
            self._pulse_plan = ((L + p['pulse_loss_m']) / p['pulse_v'], sign * p['pulse_v'])
        dur, v = self._pulse_plan
        self._pulse = (t + dur, v)
        self._pulse_plan = None
        self.state = PULSE

    def _clear_ahead(self, x, y, a, L):
        m = self.replanner.map
        s = np.arange(0.1, L + 1e-9, 0.05)
        c = m.clearance_at(x + s * math.cos(a), y + s * math.sin(a))
        return bool((c > 0.5 * m.p['inflate_m']).all())

    def _align_to(self, t, yaw, target, after):
        self._scan_queue = [target]
        self._scan_target = None
        self._after_align = after
        self.state = ALIGN

    # -- the step -----------------------------------------------------------

    def update(self, t, x, y, yaw, rate=0.0, reflex_idle=True) -> Step:
        p = self.p
        if self.done:
            return Step(None, (0.0, 0.0), False, 'done', 'mission complete')
        if not self.visited or np.hypot(*(self.visited[-1] - (x, y))) > 0.5:
            self.visited.append(np.array([x, y]))
        g = self.goal

        if self.state == ARRIVED:
            return Step(None, (0.0, 0.0), True, ARRIVED, self._arrived_note)

        if self.state == BACKOFF:
            if t < self._backoff_until:
                return Step(None, (p['backoff_v'], 0.0), False, BACKOFF, 'backing off')
            self._backoff_until = None
            self._start_stop(t)

        if self.state == SCAN:
            if self._scan_target is None and not self._scan_queue and self._scan_done_t is None:
                self._scan_seen = []
                self._scan_windows = {}
                self._plan_scan(yaw, self._full_scan_pending)
                self._full_scan_pending = False
            if self._scan_done_t is None:
                cmd = self._scan_step(t, yaw, rate)
                if self._scan_phase == 'settle' and self._scan_t0 is not None:
                    self._scan_windows[self._scan_t0] = t
                if cmd is not None:
                    return Step(None, cmd, False, SCAN, 'scanning')
                self._scan_done_t = t
            if t < self._scan_done_t + p['scan_tail_s']:
                return Step(None, (0.0, 0.0), False, SCAN, 'scan: last frames')
            self._scan_done_t = None
            best = self._scan_pick()
            if best is not None:
                self._lost_scans = 0
                self.notes.append(f'{t:.1f} {g.name}: scene seen at {math.degrees(_wrap(best.yaw - yaw)):+.0f} deg '
                                  f'({best.h.inliers} inliers, parallax {best.h.parallax_px:.0f} px)')
                self._align_to(t, yaw, best.yaw, MEASURE)
            else:
                self._lost_scans += 1
                self._last_scan_xy = np.array([x, y])
                if not self._pick_hop(t, x, y, yaw):
                    self._full_scan_pending = True
                    self.notes.append(f'{t:.1f} explore: no open direction; scanning again')
                    return Step(None, (0.0, 0.0), False, SCAN, 'no open direction')
                self.state = EXPLORE
                self.notes.append(f'{t:.1f} {g.name}: not seen; exploring')

        if self.state == ALIGN:
            cmd = self._scan_step(t, yaw, rate)
            if cmd is not None:
                return Step(None, cmd, False, ALIGN, 'aligning')
            if self._after_align == ARRIVED:
                self.state = ARRIVED
                self.notes.append(f'{t:.1f} arrived {self._arrived_note}')
                return Step(None, (0.0, 0.0), True, ARRIVED, self._arrived_note)
            if self._after_align == PULSE and self._pulse_plan is not None:
                self._start_pulse(t)
            else:
                self._start_stop(t, after_hop=self._after_align == MEASURE_AFTER_HOP)

        if self.state == PULSE:
            if t < self._pulse[0]:
                return Step(None, (self._pulse[1], 0.0), False, PULSE, 'pulse')
            self._pulse = None
            self._hop_end(t, yaw)
            return Step(None, (0.0, 0.0), False, self.state, 'pulse done')

        if self.state == MEASURE:
            if t < self._stop_t0 + p['settle_s']:
                return Step(None, (0.0, 0.0), False, MEASURE, 'settling')
            waited = t - self._stop_t0 - p['settle_s']
            if len(self._stop_meas) < p['measure_n'] and waited < p['measure_timeout_s']:
                return Step(None, (0.0, 0.0), False, MEASURE, 'measuring')
            if self.wants_motion_check and waited < p['measure_timeout_s'] + p['motion_timeout_s']:
                return Step(None, (0.0, 0.0), False, MEASURE, 'checking it moved')
            if self._motion_wanted and self._motion_px is not None and self._hop_from is not None:
                hx, hy, hyaw, ha = self._hop_from
                odom = math.hypot(x - hx, y - hy)
                if odom >= p['still_min_odom_m'] and self._motion_px < p['still_px']:
                    # the wheels turned, the view did not change: pinned against something
                    self._motion_wanted = False
                    self.stucks += 1
                    self._mark_ahead(x, y, ha)
                    self._no_progress = p['no_progress_stops']      # the next hop is a detour
                    self._backoff_until = t + p['backoff_s']
                    self.state = BACKOFF
                    self.notes.append(f'{t:.1f} stuck: odometry {odom:.2f} m, view moved {self._motion_px:.1f} px; '
                                      'backing off, marked ahead, detour next')
                    return Step(None, (p['backoff_v'], 0.0), False, BACKOFF, 'stuck')
            self._motion_wanted = False
            self.stops += 1
            med, n = self._median(self._stop_meas)
            if med is None:
                self.notes.append(f'{t:.1f} {g.name}: lost ({n} frames, no match); scanning')
                self.state = SCAN
                self._full_scan_pending = self._lost_scans > 0 or self._last is None
                self._scan_queue, self._scan_target = [], None
                self._lost_scans += 1
                return Step(None, (0.0, 0.0), False, SCAN, 'lost')
            m, par, inl = med
            self._last = m
            zooms = [q.h.zoom for q in self._stop_meas if q.h.ok and q.h.zoom == q.h.zoom]
            lz = math.log(float(np.median(zooms))) if zooms else 0.0
            if par <= p['arrive_parallax_px'] and inl >= p['arrive_min_inliers'] \
                    and abs(lz) > p['arrive_max_log_zoom']:
                # the view lines up but everything is smaller (behind the goal) or bigger (past it):
                # a far scene, where the parallax says little. Step along the photo's heading.
                a = m.yaw + m.h.yaw
                sign = 1.0 if lz < 0 else -1.0
                self._hop_from = (x, y, yaw, a)
                self.notes.append(f'{t:.1f} parallax {par:.0f} px but scales {math.exp(lz):.3f}: '
                                  f'{"forward" if sign > 0 else "back"} {p["zoom_step_m"]:.2f} m')
                if sign > 0 and abs(_wrap(a - yaw)) > math.radians(10.0):
                    self._pulse_plan = ((p['zoom_step_m'] + p['pulse_loss_m']) / p['pulse_v'], p['pulse_v'])
                    self._align_to(t, yaw, a, PULSE)
                    return Step(None, (0.0, 0.0), False, ALIGN, 'zoom step')
                self._start_pulse(t, p['zoom_step_m'], sign)
                return Step(None, (0.0, 0.0), False, PULSE, 'zoom step')
            if par <= p['arrive_parallax_px'] and inl >= p['arrive_min_inliers']:
                self._arrived_note = f'{g.name}: parallax {par:.0f} px, {inl} inliers'
                if p['align_heading'] and abs(m.h.yaw) > math.radians(p['align_tol_deg']):
                    self._align_to(t, yaw, m.yaw + m.h.yaw, ARRIVED)
                    return Step(None, (0.0, 0.0), False, ALIGN, 'final alignment')
                self.state = ARRIVED
                self.notes.append(f'{t:.1f} arrived {self._arrived_note}')
                return Step(None, (0.0, 0.0), True, ARRIVED, self._arrived_note)
            if par < p['progress_frac'] * self._best_parallax:
                self._best_parallax = par
                self._no_progress = 0
            elif par >= p['progress_min_px']:
                self._no_progress += 1
            self._plan_hop(t, x, y, yaw, m, par)
            if self.state != HOP:           # a pulse, or turning to one: no route to follow
                return Step(None, (0.0, 0.0), False, self.state, 'hop planned')

        # HOP and EXPLORE drive the route; the planner and the recovery keep it clear
        self.s_proj = igm.project_forward(self.pts, self.cum, x, y, self.s_proj)
        goal_xy = self.pts[-1]
        note = ''
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
        gave_up = rnote.startswith('[recover]') and ('budget spent' in rnote or 'no reachable' in rnote
                                                     or 'exploration off' in rnote)
        near_end = float(self.cum[-1]) - self.s_proj < (0.25 if self.state == HOP else p['hop_done_m'])
        if self.state == HOP and (near_end or gave_up):
            self._hop_end(t, yaw)
            return Step(None, (0.0, 0.0), False, self.state, 'hop done')
        if self.state == EXPLORE and self._sighted is not None:
            m, self._sighted = self._sighted, None
            self._lost_scans = 0
            self.notes.append(f'{t:.1f} {g.name}: scene seen while exploring ({m.h.inliers} inliers); stopping there')
            self._align_to(t, yaw, m.yaw, MEASURE)
            return Step(None, (0.0, 0.0), False, ALIGN, 'sighted')
        since = np.hypot(*(np.array([x, y]) - self._last_scan_xy)) if self._last_scan_xy is not None else 0.0
        if self.state == EXPLORE and (near_end or gave_up or since >= p['scan_every_m']):
            self.state = SCAN
            self._full_scan_pending = True
            self._scan_queue, self._scan_target = [], None
            return Step(None, (0.0, 0.0), False, SCAN, 'explore hop done')
        return Step(self._carrot(), None, False, self.state, note, self.pts)

    def _hop_end(self, t, yaw):
        m = self._last
        if self.p['align_heading'] and m is not None:
            target = m.yaw + m.h.yaw
            if abs(_wrap(target - yaw)) > math.radians(self.p['view_tol_deg']):
                self._align_to(t, yaw, target, MEASURE_AFTER_HOP)
                return
        self._start_stop(t, after_hop=True)

    # -- explore, bounded and biased -----------------------------------------

    def _explore_dir(self, yaw):
        if self._hint is not None:
            return np.array([math.cos(self._hint), math.sin(self._hint)])
        return super()._explore_dir(yaw)

    def _pick_hop(self, t, x, y, yaw, choice=None):
        if np.hypot(x - self.start[0], y - self.start[1]) > self.p['explore_radius_m']:
            # outside the arena's likely extent: head back towards the start
            a = math.atan2(self.start[1] - y, self.start[0] - x)
            L = min(self.p['explore_hop_m'], float(np.hypot(x - self.start[0], y - self.start[1])))
            self._set_route([(x, y), (x + L * math.cos(a), y + L * math.sin(a))])
            self._hop_t0, self._repick = t, False
            self.notes.append(f'{t:.1f} explore: {self.p["explore_radius_m"]:.0f} m from the start; back towards it')
            return True
        ok = super()._pick_hop(t, x, y, yaw, choice)
        if ok and self.pts is not None:
            end = self.pts[-1]
            r = float(np.hypot(end[0] - self.start[0], end[1] - self.start[1]))
            if r > self.p['explore_radius_m']:
                # clip the hop at the radius: keep the part of the route inside it
                d = np.hypot(self.pts[:, 0] - self.start[0], self.pts[:, 1] - self.start[1])
                keep = self.pts[d <= self.p['explore_radius_m']]
                if len(keep) >= 2:
                    self._set_route(keep)
                else:
                    self._set_route([(x, y), (x, y)])
        return ok
