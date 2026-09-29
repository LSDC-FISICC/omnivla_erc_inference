"""loop_patrol + the NYU map + the goal images: the pieces the image mission depends on.

Runs without ROS (test/indoor_sim.py and test/obstacles.py for the world). The goal-image
test needs erc_perception on the path (the sibling source tree is tried).
"""
import math
import os
import sys

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, '..', '..', 'erc_perception'))

from erc_inference.image_goal_mission import Goal  # noqa: E402
from erc_inference.loop_patrol import ARRIVED, GOTO, KnownMap, LoopPatrolMission, WallLocalizer  # noqa: E402

TRACK = yaml.safe_load(open(os.path.join(PKG, 'config', 'indoor_nyu_track.yaml')))
GOALS_FILE = os.path.join(PKG, 'config', 'indoor_nyu_goals.yaml')


def known():
    return KnownMap.from_yaml(TRACK)


def test_goal_images_classify_in_pdf_order():
    cv2 = pytest.importorskip('cv2')
    try:
        from erc_perception.cones import classify_goal_image
    except ImportError:
        pytest.skip('erc_perception not importable')
    doc = yaml.safe_load(open(GOALS_FILE))
    base = os.path.dirname(GOALS_FILE)
    got = []
    for g in doc['goals']:
        img = cv2.cvtColor(cv2.imread(os.path.join(base, g['image'])), cv2.COLOR_BGR2RGB)
        got.append(classify_goal_image(img)[0])
    # CP1 red, CP2 blue, CP3 green, CP4 yellow, Finish orange (red and orange are one class)
    assert got == ['red_orange', 'blue', 'green', 'yellow', 'red_orange']
    assert [bool(g.get('start_cone', False)) for g in doc['goals']] == [False] * 4 + [True]


def test_ring_is_the_closed_loop_without_the_spur():
    km = known()
    assert 100.0 < km.ring_len < 120.0
    assert np.allclose(km.ring[0], km.ring[-1])
    # the start spur (x < 9.1 on the south corridor) is not on the ring
    assert km.ring[:, 0].min() >= 9.0
    # a point on the spur projects to the SW corner, never to two places
    s = km.ring_s(3.0, -0.2)
    p = np.array([np.interp(s, km.ring_cum, km.ring[:, 0]), np.interp(s, km.ring_cum, km.ring[:, 1])])
    assert np.hypot(*(p - (9.1, -0.2))) < 0.3


def test_known_walls_and_corridors():
    km = known()
    assert km.inside(20.0, -0.2) and km.inside(36.2, 12.0) and km.inside(20.0, 27.8) and km.inside(9.1, 14.0)
    assert not km.inside(20.0, 14.0)            # the courtyard
    assert km.wall_reliable.sum() >= 8 and not km.wall_reliable.all()   # the curved wall is not matched


@pytest.mark.parametrize('X,Y,TH', [(9.1, 20.0, -math.pi / 2), (36.2, 8.0, math.pi / 2), (25.0, 27.8, math.pi)])
def test_wall_localizer_removes_lateral_and_heading_error(X, Y, TH):
    import indoor_sim as isim
    import obstacles as obs
    km = known()
    walls = isim.nyu_walls()
    rng = np.random.default_rng(0)
    wl = WallLocalizer(km)
    n = np.array([-math.sin(TH), math.cos(TH)])
    raw = (X + 0.4 * n[0], Y + 0.4 * n[1], TH + math.radians(6.0))
    for k in range(40):
        st = 0.08 * k
        tx, ty = X + st * math.cos(TH), Y + st * math.sin(TH)
        rx, ry = raw[0] + st * math.cos(raw[2]), raw[1] + st * math.sin(raw[2])
        b, f, cap = obs.profile(tx, ty, TH, walls, rng)
        cx, cy, cth = wl.apply(rx, ry, raw[2])
        wl.observe(cx, cy, cth, b, f, f < cap - 1e-3)
    cx, cy, cth = wl.apply(rx, ry, raw[2])
    assert abs(float(np.array([cx - tx, cy - ty]) @ n)) < 0.12
    assert abs(math.degrees(math.remainder(cth - TH, 2 * math.pi))) < 2.5


def test_wall_localizer_blind_on_the_south_centerline():
    """Known limit: the south corridor's straight outer wall is 2.15 m from its centerline,
    beyond what the 60 deg / 3 m profile sees; the curved inner wall is never matched. So no
    correction there from the middle -- only near its walls, its ends and the corners."""
    import indoor_sim as isim
    import obstacles as obs
    km = known()
    wl = WallLocalizer(km)
    b, f, cap = obs.profile(22.0, -0.2, 0.0, isim.nyu_walls(), np.random.default_rng(0))
    assert wl.observe(22.0, -0.2, 0.0, b, f, f < cap - 1e-3) is None


def test_wall_localizer_holds_still_while_turning():
    import indoor_sim as isim
    import obstacles as obs
    km = known()
    wl = WallLocalizer(km)
    b, f, cap = obs.profile(9.1, 20.0, -math.pi / 2, isim.nyu_walls(), np.random.default_rng(0))
    assert wl.observe(9.5, 20.0, -math.pi / 2 + 0.1, b, f, f < cap - 1e-3, turn_rate_dps=25.0) is None
    assert (wl.dx, wl.dy, wl.dyaw) == (0.0, 0.0, 0.0)


def mission():
    goals = [Goal('CP1', 'red_orange', 1), Goal('CP2', 'blue', 2), Goal('CP3', 'green', 3),
             Goal('Finish', 'red_orange', 4, is_start_cone=True)]
    return LoopPatrolMission(goals, known(), (0.0, 0.0), 'red_orange', {'inflate_m': 0.35}, None)


def test_start_cone_is_not_cp1():
    m = mission()
    for _ in range(3):              # the orange start cone, right behind the rover
        m.observe_cone(0.0, 0.0, 0.0, 'red_orange', 180.0, 0.8, t=0.0)
    m.update(0.5, 0.0, 0.0, 0.0)
    assert m.target is None or m.target.name != 'start'


def test_later_cone_is_mapped_now_and_visited_when_its_turn_comes():
    """CP3 seen before CP2: mapped, not visited; after CP2 the rover goes to it directly."""
    m = mission()
    t = 0.0
    # CP1 right ahead: goes there
    for k in range(3):
        m.observe_cone(20.0, -0.2, 0.0, 'red_orange', 0.0, 3.0, t=t + 0.1 * k)
    m.update(1.0, 20.0, -0.2, 0.0)
    assert m.target is not None and np.hypot(*(m.target.estimate() - (23.0, -0.2))) < 0.3
    # on the way, CP3 (green) comes into view further on: mapped, but CP1 stays the target
    for k in range(3):
        m.observe_cone(20.0, -0.2, 0.0, 'green', 5.0, 8.0, t=1.2 + 0.1 * k)
    step = m.update(2.0, 20.0, -0.2, 0.0)
    assert step.state == GOTO and m.goal.name == 'CP1'
    assert any(c.cls == 'green' for c in m.cones.clusters)
    # arrive at CP1, confirm
    step = m.update(3.0, 22.0, -0.2, 0.0)
    assert step.arrived and step.state == ARRIVED
    m.confirm(3.5, True)
    assert m.goal.name == 'CP2'
    # CP2 (blue) appears; reach it; confirm
    for k in range(3):
        m.observe_cone(23.0, -0.2, 0.0, 'blue', 0.0, 2.5, t=4.0 + 0.1 * k)
    m.update(5.0, 23.0, -0.2, 0.0)
    assert m.target.cls == 'blue'
    assert m.update(6.0, 24.5, -0.2, 0.0).arrived       # blue at 25.5, green at ~28.0
    m.confirm(6.5, True)
    # CP3: already on the map -> straight to GOTO, no search needed
    step = m.update(7.0, 24.5, -0.2, 0.0)
    assert m.goal.name == 'CP3' and m.target is not None and m.target.cls == 'green'
    assert step.state == GOTO and not step.arrived


def _stuck(m, ahead_m):
    t, step = 0.0, None
    b = np.linspace(-30, 30, 25)
    r = np.full(25, 3.0) if ahead_m is None else np.where(np.abs(b) <= 20, ahead_m, 3.0)
    hit = r < 2.95
    while t < 14.0 and m.stucks == 0:     # "driving" but not moving
        m.observe_scan(t, 19.5, -0.2, 0.0, b, r, hit)
        step = m.update(t, 19.5, -0.2, 0.0)   # between the south scan points (16, 23)
        t += 1.0 / 3.0
    return t, step


def test_stuck_backs_off_and_marks_what_it_sees_ahead():
    m = mission()
    t, step = _stuck(m, 0.6)
    assert m.stucks == 1 and t < 13.0
    assert step.command is not None and step.command[0] < 0.0      # backing off
    assert m.replanner.map.occupied()[m.replanner.map.cell(20.1, -0.2)]


def test_stuck_with_nothing_ahead_backs_off_without_marking():
    """Marking blind filled 2 m corridors with fake obstacles until no path was left."""
    m = mission()
    t, step = _stuck(m, None)
    assert m.stucks == 1
    assert step.command is not None and step.command[0] < 0.0
    assert not m.replanner.map.occupied()[m.replanner.map.cell(20.1, -0.2)]
