"""local_planner goal mode (unknown routes): the map carried between legs, A* to the checkpoint round a seen
wall, and the no-progress mark. No ROS."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from erc_controller import local_planner as lp  # noqa: E402

GOAL = dict(goal_mode=True, keep_map=True, unseen_extend_m=0.1, rejoin_respects_unseen=True)


def wall(m, x0, y0, x1, y1):
    n = int(max(abs(x1 - x0), abs(y1 - y0)) / 0.05) + 1
    r, c = m.cell(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    ok = m._inside(r, c)
    m.L[r[ok], c[ok]] = m.p['max_logodds']
    m.seen[r[ok], c[ok]] = True
    m._dist = None


def cum(pts):
    pts = np.asarray(pts, float)
    return pts, np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])


def test_map_is_carried_to_the_next_leg_in_its_frame():
    a = lp.LocalReplanner([(0, 0), (10, 0)], **GOAL)
    wall(a.map, 5, -2, 5, 2)
    # next leg's origin is (3, 1) in a's frame: the wall is at x = 2 there
    b = lp.LocalReplanner([(0, 0), (6, 0)], previous=a, offset=(3.0, 1.0), **GOAL)
    assert float(b.map.clearance_at(2.0, -1.0)) < 0.15
    assert float(b.map.clearance_at(-1.0, -1.0)) > 2.0


def test_goal_mode_plans_round_a_seen_wall_to_the_checkpoint():
    pts, c = cum([(0, 0), (12, 0)])
    r = lp.LocalReplanner(pts, **GOAL)
    wall(r.map, 6, -8, 6, 3)                       # the way round is north of y = 3
    # the route ahead (within check_ahead_m) crosses the wall: the plan to the checkpoint replaces it
    new, note = r.check(1.0, 2.0, 0.0, pts, c, 2.0)
    assert new is not None and 'route blocked' in note
    assert new[-1][0] == 12 and new[-1][1] == 0
    assert max(p[1] for p in new) > 3.0            # goes round the end of the wall
    assert float(np.min(r.map.clearance_at(new[:, 0], new[:, 1]))) > r.p['inflate_m'] - 1e-6


def test_no_progress_marks_ahead_and_replans():
    pts, c = cum([(0, 0), (12, 0)])
    r = lp.LocalReplanner(pts, **dict(GOAL, stuck_window_s=4.0))
    r.observe(0.0, 0.0, 0.0, 0.0, np.array([0.0]), np.array([3.0]), np.array([False]))
    note = ''
    for k in range(10):
        _new, n = r.check(float(k), 0.0, 0.0, pts, c, 0.0)
        note = n or note
    assert r.stuck_marks >= 1 and 'no progress' in note
    assert r.map.clearance_at(0.6, 0.0).item() <= r.p['inflate_m']


def test_rejoin_mode_unchanged_by_default():
    assert lp.DEFAULTS['goal_mode'] is False and lp.DEFAULTS['keep_map'] is False
