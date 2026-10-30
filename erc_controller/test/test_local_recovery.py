"""local_recovery.BlockedRecovery and pick_frontier, without ROS."""
import math

import numpy as np

from erc_controller import local_planner as lp
from erc_controller import local_recovery as lr


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


def test_detour_look_turns_to_the_checkpoint_then_takes_the_shorter_way_it_saw():
    # 30-oct, top of the Galileo ramp: a 33 m detour for 14 m straight, the checkpoint 80 deg to the left
    rec = lr.BlockedRecovery(look=False, explore=True, detour_look=True)
    goal = (0.0, 14.0)                                   # north; the rover faces east (yaw 0)
    detour = np.array([[0.0, 0.0], [-10.0, -5.0], [5.0, 20.0], [0.0, 14.0]])
    note = rec.detour_look(0.0, 0.0, 0.0, 0.0, goal, detour)
    assert 'looking toward the checkpoint' in note and rec.active
    # after the look the planner finds the direct way: it replaces the detour
    direct = np.array([[0.0, 0.0], [0.0, 14.0]])
    t, yaw, got = 0.0, 0.0, None
    planner = _Planner((direct, '[local-plan] goal: 19.0 m shorter; 2 points, 14.0 m to the checkpoint'))
    while t < 40.0 and rec.active:
        cmd, new_pts, _n = rec.step(t, 0.0, 0.0, yaw, planner, detour, np.array([0.0, 11.2, 40.0, 47.8]), 0.0, goal)
        got = new_pts if new_pts is not None else got
        if cmd is not None:
            yaw += cmd[1] * 0.1
        t += 0.1
    assert not rec.active and got is not None and np.allclose(got[-1], goal)
    # it turned left, toward the checkpoint; it stops commanding rate x lead_s early (the rover's delay carries
    # it the rest of the way), and this integration has no delay
    assert 45.0 < math.degrees(yaw) < 100.0
    # not again from the same place
    assert rec.detour_look(t + 20.0, 0.5, 0.0, 0.0, goal, detour) == ''


def test_detour_look_ignores_short_detours_and_a_goal_ahead():
    rec = lr.BlockedRecovery(explore=True, detour_look=True)
    goal = (14.0, 0.0)
    assert rec.detour_look(0.0, 0.0, 0.0, 0.0, goal, np.array([[0, 0], [7, 3], [14, 0]])) == ''   # 1.1x
    long_ahead = np.array([[0.0, 0.0], [5.0, 10.0], [14.0, 10.0], [14.0, 0.0]])
    assert rec.detour_look(0.0, 0.0, 0.0, 0.0, goal, long_ahead) == ''                           # ahead


def test_backoff_backs_away_from_a_pinned_nose_then_waits():
    b = lr.BackOff()
    t, out = 0.0, []
    while t < 12.0:
        b.profile(t, np.array([-10.0, 0.0, 10.0]), np.array([0.3, 0.25, 0.3]))
        v, _n = b.step(t, 0.0)                           # the drive is turning in place, not moving forward
        out.append((t, v))
        t += 0.1
    backing = [tt for tt, v in out if v is not None and v < 0]
    assert backing and abs(backing[0] - 4.0) < 0.15      # after after_s pinned
    assert 2.8 < backing[-1] - backing[0] < 3.1          # for s
    assert b.backs == 1                                  # retry_s holds the next one


def test_backoff_leaves_a_rover_that_drives_forward_alone():
    b = lr.BackOff()
    for k in range(100):
        b.profile(0.1 * k, np.array([0.0]), np.array([0.3]))
        assert b.step(0.1 * k, 0.2)[0] is None


def test_turnback_look_looks_to_both_sides_goal_side_first():
    rec = lr.BlockedRecovery(explore=True, turnback_look=True)
    # rover faces east; the new route starts by going back west; the checkpoint is north (left)
    back = np.array([[0.0, 0.0], [-5.0, 0.0], [-5.0, 10.0]])
    note = rec.detour_look(0.0, 0.0, 0.0, 0.0, (0.0, 10.0), back)
    assert 'turns back' in note and 'left first' in note and rec.active
    assert len(rec._queue) == 2 and rec._queue[0] > 0 > rec._queue[1]
    # a route ahead does not trigger it
    rec2 = lr.BlockedRecovery(explore=True, turnback_look=True)
    assert rec2.detour_look(0.0, 0.0, 0.0, 0.0, (0.0, 10.0), np.array([[0, 0], [5, 0], [5, 10]])) == ''


def test_backoff_ignores_a_post_at_the_side_and_stops_after_its_tries():
    b = lr.BackOff()
    for k in range(80):                                  # a post in one bin of nine, pinned: not a wall across
        b.profile(0.1 * k, np.linspace(-20, 20, 9), np.r_[[0.3], np.full(8, 3.0)])
        assert b.step(0.1 * k, 0.0, 0.0, 0.0)[0] is None
    b = lr.BackOff(retry_s=0.5)
    t, backs = 0.0, 0
    while t < 60.0:                                      # a wall across, the rover led back to it every time
        b.profile(t, np.linspace(-20, 20, 9), np.full(9, 0.3))
        v, _n = b.step(t, 0.0, 0.2, 0.0)
        t += 0.1
    assert b.backs == 2                                  # tries, then the replanning has to do it
