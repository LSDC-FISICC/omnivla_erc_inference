"""People in the plant mission (16-oct): kept off the map (people_mask.py), and one in the way is
waited for until gone and the profile has looked again (explore_search._people_step).
Run by hand: python3 -m pytest test/test_people.py"""
import math

import numpy as np

from erc_inference.explore_search import ExploreSearchMission
from erc_inference.image_goal_mission import Goal
from erc_inference.loop_patrol import LOCAL_INDOOR, PATROL_COMPETITION

BEARINGS = np.arange(-60.0, 61.0, 5.0)
OPEN = np.full(len(BEARINGS), 3.0)                 # an empty corridor ahead


def mission(**kw):
    p = dict(PATROL_COMPETITION, start_scan=False, count_on_sight=True, forward_first=True, look_around=False,
             drive='reactive')
    p.update(kw)
    return ExploreSearchMission([Goal('Maceta', 'potted plant', 1)], (0.0, 0.0), dict(LOCAL_INDOOR), **p)


def with_person(ranges, right, left, rng):
    """The profile with a person standing between bearings right..left at rng."""
    r = ranges.copy()
    r[(BEARINGS >= right - 2.5) & (BEARINGS <= left + 2.5)] = rng
    return r


def occupied_near(m, x, y, radius=0.5):
    rows, cols = m.map.cell(np.array([x]), np.array([y]))
    k = int(math.ceil(radius / m.map.p['resolution_m']))
    win = m.map.occupied()[rows[0] - k:rows[0] + k + 1, cols[0] - k:cols[0] + k + 1]
    return int(win.sum())


def run(m, t0, t1, profile, person=None, dt=1 / 3.0):
    """Profiles and (YOLO) sightings at ~3 Hz from the start pose; -> the last step and the steps."""
    steps, t = [], t0
    while t < t1:
        if person is not None:
            m.observe_people(t, 0.0, 0.0, 0.0, [person])
        m.observe_scan(t, 0.0, 0.0, 0.0, BEARINGS, profile, profile < 2.95)
        steps.append(m.update(t, 0.0, 0.0, 0.0, 0.0))
        t += dt
    return steps[-1], steps


def test_a_person_ahead_is_not_written_into_the_map():
    m = mission()
    run(m, 0.0, 3.0, with_person(OPEN, -8.0, 8.0, 1.6), person=(-8.0, 8.0, 1.6))
    assert occupied_near(m, 1.6, 0.0) == 0
    # ...the same profile with nobody seen: a wall there, as before
    m2 = mission(people_wait=False)
    run(m2, 0.0, 3.0, with_person(OPEN, -8.0, 8.0, 1.6))
    assert occupied_near(m2, 1.6, 0.0) > 0


def test_hits_written_before_the_person_was_recognised_are_dropped():
    m = mission()
    run(m, 0.0, 2.0, with_person(OPEN, -8.0, 8.0, 1.6))                 # YOLO has not seen them yet
    assert occupied_near(m, 1.6, 0.0) > 0
    m.observe_people(2.1, 0.0, 0.0, 0.0, [(-8.0, 8.0, 1.6)])
    assert occupied_near(m, 1.6, 0.0) == 0


def test_it_waits_for_a_person_in_the_way_and_drives_on_once_they_are_gone_and_the_floor_is_seen():
    m = mission()
    step, steps = run(m, 0.0, 4.0, with_person(OPEN, -8.0, 8.0, 1.6), person=(-8.0, 8.0, 1.6))
    assert step.command == (0.0, 0.0) and 'waiting' in step.note
    # gone: still for person_gone_s, then two profiles, then on
    step, steps = run(m, 4.0, 7.0, OPEN)
    waits = [s.command == (0.0, 0.0) and 'waiting' in s.note for s in steps]
    assert waits[0] and not waits[-1]
    assert steps[-1].command[0] > 0.0
    assert m.people_waits and m.people_waits[-1][2] == 'gone'
    assert occupied_near(m, 1.6, 0.0) == 0


def test_a_person_beside_the_corridor_or_far_is_not_waited_for():
    m = mission()
    _s, steps = run(m, 0.0, 3.0, with_person(OPEN, 35.0, 45.0, 2.0), person=(35.0, 45.0, 2.0))   # 1.2-1.4 m left
    assert not any('waiting' in s.note for s in steps)
    m = mission()
    _s, steps = run(m, 0.0, 3.0, OPEN, person=(-3.0, 3.0, 6.0))                                  # 6 m ahead
    assert not any('waiting' in s.note for s in steps)


def test_someone_who_stays_is_driven_round_after_the_longest_wait():
    m = mission(person_wait_max_s=5.0)
    _s, steps = run(m, 0.0, 8.0, with_person(OPEN, -8.0, 8.0, 1.6), person=(-8.0, 8.0, 1.6))
    assert any('waiting' in s.note for s in steps[:10])
    assert not any('waiting' in s.note for s in steps[-6:])
    assert m.people_waits[-1][2] == 'timeout'


def test_off_people_wait_keeps_them_off_the_map_but_never_waits():
    m = mission(people_wait=False)
    _s, steps = run(m, 0.0, 3.0, with_person(OPEN, -8.0, 8.0, 1.6), person=(-8.0, 8.0, 1.6))
    assert not any('waiting' in s.note for s in steps)
    assert occupied_near(m, 1.6, 0.0) == 0


def test_one_frame_with_someone_in_the_way_does_not_stop_the_rover():
    m = mission()
    m.observe_people(0.0, 0.0, 0.0, 0.0, [(-8.0, 8.0, 1.6)])          # once, then never again
    _s, steps = run(m, 0.0, 3.0, OPEN)
    assert not any('waiting' in s.note for s in steps)
