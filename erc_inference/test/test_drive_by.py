"""What the drive-past-and-count mission and its MPPI follower rest on (12-oct): the opening ahead
on the profile of the real corridor, a tilt that needs the accelerometer to agree, the in-place
dead zone inverted. Run by hand: python3 -m pytest test/test_drive_by.py"""
import math

import numpy as np

from erc_inference.explore_search import ExploreSearchMission
from erc_inference.image_goal_mission import Goal
from erc_inference.loop_patrol import LOCAL_INDOOR, PATROL_COMPETITION, SAFETY_INDOOR
from erc_inference.nav2_route_follower_node import predict_pose, rest_rate, snap_command
from erc_inference.safety_envelope import DEFAULTS as SAFETY_DEFAULTS
from erc_inference.safety_envelope import SafetyEnvelope

BEARINGS = np.arange(-60.0, 61.0, 5.0)
# /erc/free_space 1 s into rosbags/mission_1012_1429_macetas, right to left: a 2.6 m corridor
CORRIDOR = np.array([3.0, 3.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.1, 2.6, 3.0, 3.0, 3.0, 3.0, 3.0, 2.9, 2.3, 1.9, 1.6,
                     1.5, 1.5, 1.5, 1.4, 3.0, 3.0, 3.0])


def mission(**kw):
    p = dict(PATROL_COMPETITION, start_scan=False, count_on_sight=True, forward_first=True, look_around=False)
    p.update(kw)
    return ExploreSearchMission([Goal('Maceta', 'potted plant', 1)], (0.0, 0.0), dict(LOCAL_INDOOR), **p)


def drive(m, ranges, seconds, x=0.0):
    step = None
    for k in range(int(seconds * 3)):
        t = k / 3.0
        m.observe_scan(t, x, 0.0, 0.0, BEARINGS, ranges, ranges < 2.95)
        step = m.update(t, x, 0.0, 0.0, 0.0)
    return step


def test_open_corridor_is_driven_straight_on_without_turning_first():
    m = mission()
    step = drive(m, CORRIDOR, 2.0)
    assert step.command is None and step.carrot is not None          # a route, not a scan
    assert abs(math.degrees(math.atan2(step.carrot[1], step.carrot[0]))) < 10.0
    assert any('open ahead' in n for n in m.notes)


def test_someone_standing_in_front_closes_the_way_and_nothing_turns():
    r = CORRIDOR.copy()
    r[9:15] = 0.6
    m = mission()
    step = drive(m, r, 4.0)
    assert m._opening(0.0) is None
    assert step.command in (None, (0.0, 0.0))                        # a way round by the map, or standing still
    assert not any('looking around' in n for n in m.notes)


def test_a_plant_counts_once_when_the_rover_drives_past_it():
    m = mission(max_shows=3)
    for k in range(90):                                              # 30 s at 0.25 m/s past a plant at (5, 1)
        t, x = k / 3.0, 0.25 * k / 3.0
        m.observe_scan(t, x, 0.0, 0.0, BEARINGS, CORRIDOR, CORRIDOR < 2.95)
        d = math.hypot(5.0 - x, 1.0)
        b = math.degrees(math.atan2(1.0, 5.0 - x))
        if abs(b) < 50.0:
            m.observe_objects(x, 0.0, 0.0, [('potted plant', b, d)], t)
        m.update(t, x, 0.0, 0.0, 0.0)
    assert m.index == 1
    assert sum('counted' in n for n in m.notes) == 1


def envelope(confirm):
    p = dict(SAFETY_DEFAULTS, **SAFETY_INDOOR)
    p['safety.tilt_accel_confirm_deg'] = confirm
    return SafetyEnvelope(), p


def lean(deg):
    """(quaternion of a roll of `deg`, the accelerometer's reading there)."""
    a = math.radians(deg)
    return (math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)), (0.0, 9.81 * math.sin(a), 9.81 * math.cos(a))


def test_filter_tilt_alone_does_not_stop_when_the_accelerometer_says_level():
    for confirm, stops in ((0.0, True), (6.0, False)):
        e, p = envelope(confirm)
        for k in range(100):                                         # level, armed, then the filter reads 17 deg
            t = k * 0.05
            q, _ = lean(0.0 if k < 30 else 17.0)
            e.observe_attitude(t, *q)
            e.observe_accel(t, *lean(0.5)[1])
            if k == 20:
                e.arm()
            tilted, _ = e._tilt_gate(t, p)
        assert tilted is stops


def test_a_real_tilt_still_stops_with_the_accelerometer_agreeing_or_far_past_the_limit():
    e, p = envelope(6.0)
    for k in range(100):
        t = k * 0.05
        q, a = lean(0.0 if k < 30 else 18.0)
        e.observe_attitude(t, *q)
        e.observe_accel(t, *a)
        if k == 20:
            e.arm()
        tilted, _ = e._tilt_gate(t, p)
    assert tilted
    e, p = envelope(6.0)
    for k in range(40):                                              # 31 deg on the filter: no waiting for the median
        t = k * 0.05
        e.observe_attitude(t, *lean(0.0 if k < 30 else 31.0)[0])
        e.observe_accel(t, *lean(0.0)[1])
        if k == 20:
            e.arm()
        tilted, _ = e._tilt_gate(t, p)
    assert tilted


def test_in_place_dead_zone_is_inverted_and_predicted():
    for want in (0.05, 0.2, 0.34, -0.3):
        v, cmd = snap_command(0.0, want, w_max=0.3, rest_deadband=0.10, rest_slope=1.7)
        assert v == 0.0 and abs(rest_rate(cmd, 0.10, 1.7) - want) < 1e-6
    assert snap_command(0.0, 0.005, rest_deadband=0.10, rest_slope=1.7) == (0.0, 0.0)
    assert snap_command(0.0, 0.2, k_w_in_place=1.18)[1] == 0.2 / 1.18             # 0: the gain, as before
    # 1 s of 0.3 rad/s commanded in place: 0.34 rad on this unit, not 0.3 x a gain
    _x, _y, yaw = predict_pose(0.0, 0.0, 0.0, [(0.0, 0.0, 0.3)], 1.0, 1.0, rest_deadband=0.10, rest_slope=1.7)
    assert abs(yaw - 0.34) < 0.01
