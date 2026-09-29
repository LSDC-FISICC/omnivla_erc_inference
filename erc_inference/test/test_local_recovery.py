"""local_recovery.BlockedRecovery and pick_frontier, without ROS."""
import math

import numpy as np

from erc_inference import local_planner as lp
from erc_inference import local_recovery as lr


class _Planner:
    """A LocalReplanner stand-in: check() answers what the test says; map is real."""

    def __init__(self, answer, cmap=None):
        self.answer = answer
        self.map = cmap
        self._last_check = self._last_plan = 0.0
        self.calls = 0

    def check(self, t, x, y, pts, cum, s_proj, extra_lethal=None):
        self.calls += 1
        return self.answer


def _route(length=20.0):
    pts = np.array([[0.0, 0.0], [length, 0.0]])
    return pts, np.array([0.0, length])


def _run(rec, planner, pts, cum, t_end=60.0, gain=1.0, dt=0.1):
    """Integrate the rover's yaw with the commands the recovery gives; -> (yaws, notes, final pts)."""
    t, yaw, yaws, notes = 0.0, 0.0, [], []
    while t < t_end:
        cmd, new_pts, note = rec.step(t, 0.0, 0.0, yaw, planner, pts, np.asarray(cum), 0.0, (20.0, 0.0))
        if note:
            notes.append(note)
        if new_pts is not None:
            pts = new_pts
            cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        if cmd is not None:
            yaw += gain * cmd[1] * dt
        yaws.append(yaw)
        if not rec.active:
            break
        t += dt
    return np.array(yaws), notes, pts


def test_nothing_happens_when_both_off():
    rec = lr.BlockedRecovery(look=False, explore=False)
    assert not rec.trigger(0.0, 0.0, 0.3, no_path=True)
    assert not rec.active


def test_look_turns_both_ways_then_hands_back():
    rec = lr.BlockedRecovery(look=True, lead_s=0.0, settle_s=0.5)
    pts, cum = _route()
    assert rec.trigger(0.0, 0.0, goal_bearing=0.2)          # goal on the left: left first
    yaws, notes, _ = _run(rec, _Planner((None, '')), pts, cum)
    assert not rec.active
    assert math.degrees(yaws.max()) > 50 and math.degrees(yaws.min()) < -50
    assert int(np.argmax(yaws)) < int(np.argmin(yaws))      # left before right
    assert any('route clear' in n or 'facing' in n for n in notes)


def test_reflex_mid_manoeuvre_blocks_the_trigger():
    rec = lr.BlockedRecovery(look=True)
    assert not rec.trigger(0.0, 0.0, 0.0, reflex_idle=False)


def test_cooldown_after_a_recovery():
    rec = lr.BlockedRecovery(look=True, lead_s=0.0, settle_s=0.2, cooldown_s=8.0)
    pts, cum = _route()
    rec.trigger(0.0, 0.0, 0.0)
    _run(rec, _Planner((None, '')), pts, cum)
    assert not rec.trigger(1.0, 0.0, 0.0)                    # still cooling down


def test_face_stops_early_by_the_measured_rate():
    """The rover keeps turning after the command is cut; FACE must stop commanding early."""
    rec = lr.BlockedRecovery(look=False, explore=True, lead_s=2.4, settle_s=0.1)
    rec._yaws = [(0.0, 0.0), (0.5, math.radians(10))]       # turning at 20 deg/s
    assert rec._rate(0.5) > math.radians(15)


def _open_map():
    """Rover at the origin facing east; a wall across the route at x=3; seen ground in front."""
    cmap = lp.LocalCostmap(-10, -10, 25, 10)
    b = np.linspace(-40, 40, 17)
    for yaw in (0.0, 0.6, -0.6):
        ranges = np.full(len(b), 2.5)
        hit = np.zeros(len(b), bool)
        for _ in range(3):
            cmap.update(0.0, 0.0, yaw, b, ranges, hit)
    for _ in range(3):                                      # the wall
        cmap.update(0.0, 0.0, 0.0, np.linspace(-15, 15, 7), np.full(7, 3.0), np.ones(7, bool))
    return cmap


def test_frontier_is_reachable_and_toward_the_goal():
    cmap = _open_map()
    f, path = lr.pick_frontier(cmap, 0.0, 0.0, (20.0, 0.0), [], lr.DEFAULTS)
    assert f is not None and path is not None
    assert 1.0 <= math.hypot(*f) <= 8.0
    assert f[0] > 0.0                                        # ahead, not behind the rover


def test_no_path_explores_once_look_is_off():
    cmap = _open_map()
    rec = lr.BlockedRecovery(look=False, explore=True, settle_s=0.1, lead_s=0.0)
    pts, cum = _route()
    assert rec.trigger(0.0, 0.0, 0.0, no_path=True)
    _yaws, notes, new_pts = _run(rec, _Planner((None, '[local-plan] blocked 3.0 m ahead; no path to route s=9'), cmap),
                                 pts, cum)
    assert rec.explores == 1
    assert any('exploring frontier 1/' in n for n in notes)
    assert new_pts.shape != pts.shape or not np.allclose(new_pts, pts)   # the route changed


def test_exploration_budget_is_spent():
    cmap = _open_map()
    rec = lr.BlockedRecovery(look=False, explore=True, settle_s=0.0, lead_s=0.0, cooldown_s=0.0,
                             explore_max=2)
    pts, cum = _route()
    planner = _Planner((None, 'no path'), cmap)
    notes = []
    for k in range(4):
        rec.trigger(10.0 * k, 0.0, 0.0, no_path=True)
        _y, n, _p = _run(rec, planner, pts, cum)
        notes += n
    assert rec.explores == 2
    assert any('budget spent' in n for n in notes)
