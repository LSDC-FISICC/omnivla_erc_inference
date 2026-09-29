"""The off-road checkpoint flags: erc_perception.flags (detector) and flag_mission (the mission).

The organisers (29-sept): three blue flags are the checkpoints, within 1 m of each. Their photo
(config/goal_images/offroad/flag_example.jpg, reduced) is the colour reference. No ROS; the
detector test needs erc_perception on the path (the sibling source tree is tried).
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, '..', '..', 'erc_perception'))

cv2 = pytest.importorskip('cv2')
flags = pytest.importorskip('erc_perception.flags')

from erc_inference.flag_mission import BACKOFF, FlagMission  # noqa: E402
from erc_inference.image_goal_mission import APPROACH, ARRIVED  # noqa: E402

EXAMPLE = os.path.join(PKG, 'config', 'goal_images', 'offroad', 'flag_example.jpg')


def background(seed=0, w=1024, h=576):
    """Rocks-and-wall-ish clutter without saturated blue."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 190, np.uint8)
    img[h // 2:] = (150, 140, 120)
    for _ in range(600):
        c = tuple(int(v) for v in rng.integers(90, 230, 3))
        c = (c[0], c[1], min(c[2], c[1]))                      # never more blue than green
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(h // 2, h))), int(rng.integers(4, 18)), c, -1)
    return img


def test_the_organisers_flag_is_found_and_the_tripod_is_not():
    img = cv2.cvtColor(cv2.imread(EXAMPLE), cv2.COLOR_BGR2RGB)
    d = flags.detect(img)
    assert len(d) == 1
    # the cloth, left of centre and in the lower half of the portrait photo
    assert 60 < d[0].u < 160 and 480 < d[0].v < 640


@pytest.mark.parametrize('x,y', [(0.6, 0.0), (1.5, 0.4), (3.0, -1.0), (6.0, 0.5)])
def test_range_and_bearing_of_a_rendered_flag(x, y):
    img = flags.render_flag(background(), x, y, height_m=0.15)
    d = flags.detect(img)
    assert len(d) == 1
    assert abs(d[0].range_m - math.hypot(x, y)) <= 0.12 * math.hypot(x, y)
    assert abs(d[0].bearing_deg - math.degrees(math.atan2(y, x))) < 1.0


def test_nothing_in_clutter_without_blue():
    assert flags.detect(background(1)) == []


# -- the mission, on scripted sightings -------------------------------------------------------

def mission(n=3, **kw):
    return FlagMission(n, (0.0, 0.0), {'inflate_m': 0.3, 'unseen_extend_m': 0.1}, None, **kw)


def sight(m, t, x, y, yaw, fx, fy, H=0.15):
    d = math.hypot(fx - x, fy - y)
    b = math.degrees(math.atan2(fy - y, fx - x) - yaw)
    ang = H / d
    m.observe_cone(x, y, yaw, 'blue', b, 0.15 / ang, t=t, ang_height=ang)


def test_a_flag_in_view_becomes_the_target_and_arrival_is_by_the_live_view():
    m = mission()
    for k in range(3):
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 3.0, 0.0)
    step = m.update(0.5, 0.0, 0.0, 0.0)
    assert m.target is not None and np.hypot(*(m.est(m.target) - (3.0, 0.0))) < 0.1
    assert step.state in (APPROACH, 'scan')
    sight(m, 1.0, 2.5, 0.0, 0.0, 3.0, 0.0)                 # standing 0.5 m from it
    step = m.update(1.1, 2.5, 0.0, 0.0)
    assert step.arrived and m.state == ARRIVED


def test_arrival_is_called_early_enough_to_stop_off_the_flag():
    """Moving at 0.25 m/s the rover goes on ~1.5 s after the stop: arrival comes ~0.4 m early."""
    m = mission()
    t, x = 0.0, 0.0
    arrived_at = None
    while t < 30.0:
        sight(m, t - 0.8, x - 0.2, 0.0, 0.0, 4.0, 0.0)     # the picture lags 0.8 s
        if m.update(t, x, 0.0, 0.0).arrived:
            arrived_at = x
            break
        x += 0.25 / 3.0
        t += 1.0 / 3.0
    assert arrived_at is not None and 4.0 - arrived_at >= 0.8   # stops ~0.4 m further on


def test_the_cloth_height_is_fitted_from_the_approach():
    m = mission()
    for k in range(12):                       # driving at it from 5 m to 2 m; true cloth 0.20 m
        x = 0.25 * k
        sight(m, 0.3 * k, x, 0.1 * math.sin(k), 0.0, 5.0, 0.3, H=0.20)
    f = m.flags[0]
    assert math.isfinite(f.height) and abs(f.height - 0.20) < 0.02
    assert abs(m.height_scale() - 0.20 / 0.15) < 0.15


def test_rejected_with_a_calibrated_range_means_the_wrong_flag():
    m = mission()
    for k in range(3):
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 2.0, 0.0)
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 2.0, 2.5)
    m.update(0.5, 0.0, 0.0, 0.0)
    first = m.target
    first.height = 0.15                       # its range scale is known
    m.state = ARRIVED
    m.confirm(1.0, False)
    assert 0 in first.wrong and m.target is None
    m.update(1.5, 0.0, 0.0, 0.0)
    assert m.target is not None and m.target is not first


def test_rejected_with_the_range_scale_assumed_pulses_closer_first():
    m = mission()
    for k in range(3):
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 0.9, 0.0)
    m.update(0.5, 0.0, 0.0, 0.0)
    m.state = ARRIVED
    m.confirm(0.6, False)
    step = m.update(0.7, 0.0, 0.0, 0.0)
    assert step.command is not None and step.command[0] > 0.0 and not step.arrived
    t = 0.7
    while not step.arrived and t < 10.0:
        t += 0.3
        step = m.update(t, 0.0, 0.0, 0.0)
    assert step.arrived                        # asks the SDK again after the pulse


def test_the_photo_decides_which_flag_is_which_checkpoint():
    m = mission()
    for k in range(3):
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 2.0, 0.0)          # nearer
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 4.0, 3.0)          # further, off to the left
    # a frame facing the far one; the photo's scene matched at its centre (both flags in view)
    m.observe_scene(0.5, 0.0, 0.0, math.atan2(3.0, 4.0), [(60, 0.0), (0, float('nan')), (0, float('nan'))])
    m.update(0.6, 0.0, 0.0, 0.0)
    assert np.hypot(*(m.est(m.target) - (4.0, 3.0))) < 0.3


def test_accepted_flags_are_obstacles_and_never_candidates_again():
    m = mission()
    for k in range(4):
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 2.0, 0.0)
    m.update(0.5, 0.0, 0.0, 0.0)
    f = m.target
    m.state = ARRIVED
    m.confirm(1.0, True)
    assert f.visited and m.goal.name == 'CP2'
    assert m.replanner.map.occupied()[m.replanner.map.cell(*f.estimate())]
    m.update(1.5, 0.0, 0.0, 0.0)
    assert m.target is None


def test_a_flag_under_the_rover_makes_it_back_off():
    m = mission()
    for k in range(4):
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 1.0, 0.0)
    step = m.update(5.0, 0.9, 0.0, 0.0)          # on it, and it is out of view
    assert step.command is not None and step.command[0] < 0.0 and step.state == BACKOFF


def test_three_flags_end_to_end_in_the_arena():
    import homing_sim
    r = homing_sim.run_flags(seed=6, t_limit=600.0)
    assert r['reached'] == 3 and all(x['err'] <= 1.0 for x in r['results'])


# -- the organisers' mechanics (29-sept): seen from <= 3 m, any order ---------------------------

def competition():
    from erc_inference.flag_mission import FLAG_COMPETITION
    return FlagMission(3, (0.0, 0.0), {'inflate_m': 0.3, 'unseen_extend_m': 0.1}, None, **FLAG_COMPETITION)


def test_competition_shows_the_nearest_flag_from_about_a_metre_and_a_half():
    m = competition()
    for k in range(5):                              # competition mode wants 4 sightings of a flag
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 1.4, 0.0)
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 4.0, 2.0)
    step = m.update(0.5, 0.0, 0.0, 0.0)
    assert step.state == 'show' and np.hypot(*(m.est(m.target) - (1.4, 0.0))) < 0.1
    t = 0.5
    while not step.arrived and t < 20.0:
        t += 0.5
        sight(m, t, 0.0, 0.0, 0.0, 1.4, 0.0)
        step = m.update(t, 0.0, 0.0, 0.0)
    assert step.arrived and t >= m.p['show_s']
    m.confirm(t, True)
    assert not m.done and len(m.goals) == 6        # keeps looking for new flags after the three


def test_a_flag_near_where_one_was_shown_is_that_one():
    m = competition()
    for k in range(5):
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 1.4, 0.0)
    step = m.update(0.5, 0.0, 0.0, 0.0)
    t = 0.5
    while not step.arrived and t < 20.0:
        t += 0.5
        sight(m, t, 0.0, 0.0, 0.0, 1.4, 0.0)
        step = m.update(t, 0.0, 0.0, 0.0)
    m.confirm(t, True)
    for k in range(5):                              # the same flag, mapped again 1.0 m off
        sight(m, t + 0.1 * k, 2.0, 1.0, math.pi, 1.4, 1.0)
    m.update(t + 1.0, 2.0, 1.0, math.pi)
    assert m.target is None


def test_competition_ignores_far_sightings_and_needs_four():
    """Indoor run (29-sept): false targets 4-12 m away filled the map."""
    m = competition()
    for k in range(6):
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 7.0, 0.0)     # beyond 6 m: ignored
    for k in range(3):
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 3.0, 2.0)     # three sightings: not yet a flag
    m.update(1.0, 0.0, 0.0, 0.0)
    assert m.target is None
    sight(m, 1.1, 0.0, 0.0, 0.0, 3.0, 2.0)
    m.update(1.2, 0.0, 0.0, 0.0)
    assert m.target is not None and np.hypot(*(m.est(m.target) - (3.0, 2.0))) < 0.2


def test_a_flag_not_reached_in_time_is_left_for_a_while():
    """Indoor run: a target behind a step was chased for minutes."""
    m = competition()
    for k in range(5):
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 4.0, 0.0)
        sight(m, 0.1 * k, 0.0, 0.0, 0.0, 1.0, 4.5)
    m.update(1.0, 0.0, 0.0, 0.0)
    first = m.target
    assert first is not None
    m.update(1.0 + m.p['target_timeout_s'] + 1.0, 0.0, 0.0, 0.0)     # still not there
    assert m.target is not first and first.skip_until > 1.0
    m.update(1.0 + m.p['target_timeout_s'] + 2.0, 0.0, 0.0, 0.0)
    assert m.target is not None and m.target is not first            # the other flag now
