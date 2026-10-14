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


# -- what the three runs of 12-oct 16:00 added: the rover climbs what it touches ----------------

def reflex_envelope():
    p = dict(SAFETY_DEFAULTS, **SAFETY_INDOOR)
    p.update({'safety.tilt_enabled': False, 'safety.obstacle_enabled': True, 'safety.stop_distance_m': 0.9,
              'safety.stop_confirm_ticks': 1, 'safety.stale_stops_turning': True, 'safety.too_close_m': 0.35})
    return SafetyEnvelope(), p


def profile(e, t, ahead):
    r = np.full(len(BEARINGS), 3.0)
    r[np.abs(BEARINGS) <= 20.0] = ahead
    e.observe_free_space(t, BEARINGS, r, 3.0)


def test_nothing_is_driven_at_within_the_stop_distance_but_it_may_turn():
    e, p = reflex_envelope()
    profile(e, 0.0, 2.0)
    assert e.limit(0.1, 0.25, 0.0, p)[0] == 0.25
    profile(e, 0.4, 0.8)
    v, w, _hard, note = e.limit(0.5, 0.25, 0.2, p)
    assert v == 0.0 and w == 0.2 and 'obstacle' in note


def test_too_close_is_straight_back_and_never_a_turn():
    e, p = reflex_envelope()
    profile(e, 0.0, 0.2)                                 # the post of mission_1012_1607, 345 s
    assert e.limit(0.1, 0.0, -0.3, p)[:2] == (-0.25, 0.0)
    assert e.limit(0.2, 0.25, 0.0, p)[:2] == (-0.25, 0.0)
    profile(e, 1.0, 0.45)                                # backed off, not yet clear of it by the margin
    assert e.limit(1.1, 0.0, -0.3, p)[:2] == (-0.25, 0.0)
    profile(e, 1.5, 0.6)
    assert e.limit(1.6, 0.0, -0.3, p)[:2] == (0.0, -0.3)   # now it may turn (not advance: 0.6 < 0.9)
    profile(e, 2.0, 0.2)
    for k in range(30):                                  # still there after too_close_backoff_s: stand still
        t = 2.1 + 0.1 * k
        profile(e, t, 0.2)
        out = e.limit(t, 0.0, -0.3, p)
    assert out[:2] == (0.0, 0.0)
    assert e.limit(5.2, -0.25, 0.0, p)[:2] == (-0.25, 0.0)   # the mission's own back-off passes


def test_with_no_fresh_profile_nothing_moves():
    e, p = reflex_envelope()
    profile(e, 0.0, 3.0)
    assert e.limit(1.0, 0.25, 0.1, p)[:2] == (0.25, 0.1)
    assert e.limit(1.6, 0.25, 0.1, p)[:2] == (0.0, 0.0)      # 1.5 s without one: the wall of 1559 came in 4.8
    assert e.limit(1.7, 0.0, 0.3, p)[:2] == (0.0, 0.0)


def test_a_band_at_1p8_m_does_not_close_the_corridor_and_turning_back_waits():
    r = CORRIDOR.copy()
    r[8:17] = 1.8                                        # mission_1012_1556, 10 s: nothing was there
    m = mission()
    step = drive(m, r, 2.0)
    assert step.carrot is not None and any('open ahead' in n for n in m.notes)


# -- reactive_drive: the mission steering itself (14-oct) -----------------------------------------

from erc_inference.reactive_drive import ReactiveDrive  # noqa: E402


def drive_once(profile_ranges, goal_deg=0.0, n=8):
    d = ReactiveDrive()
    hit = profile_ranges < 2.95
    v = w = 0.0
    for k in range(n):
        v, w = d.step(0.2 * k, 0.0, (BEARINGS, profile_ranges, hit), 0.0, goal_deg)
    return d, v, w


def test_an_open_corridor_is_driven_at_full_speed_without_turning():
    d, v, w = drive_once(CORRIDOR)
    assert v > 0.2 and abs(w) < 0.1 and d.info['mode'] == 'drive'


def test_a_pot_ahead_on_the_left_is_passed_on_the_right_without_stopping():
    # mission_1014_1405_carrot, 36-60 s: a pot 0.4-0.7 m ahead on the left, the corridor open on the
    # right. The carrot drove at it, the envelope stopped it, it backed off, and again, for 24 s.
    r = np.array([2.1, 2.4, 2.8, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                  0.7, 0.7, 0.6, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 3.0, 3.0, 3.0, 3.0])
    d, v, w = drive_once(r, goal_deg=5.0)
    assert d.info['heading'] < -10.0          # chooses the right
    assert w < 0.0 and v >= 0.10              # turns right while it keeps moving


def test_walled_in_ahead_it_turns_in_place_towards_the_room():
    r = np.full(len(BEARINGS), 0.5)
    r[BEARINGS > 45] = 3.0
    d, v, w = drive_once(r)
    assert v == 0.0 and w > 0.0


def test_the_turn_already_commanded_is_not_commanded_again():
    # 30 deg to the left: the first command turns; with that turn in flight (1.3 s), the next ones
    # ask less, though the profile still shows the opening 30 deg off
    r = np.full(len(BEARINGS), 0.8)
    r[(BEARINGS > 20) & (BEARINGS < 40)] = 3.0
    d = ReactiveDrive()
    ws = [d.step(0.2 * k, 0.0, (BEARINGS, r, r < 2.95), 0.0, 30.0)[1] for k in range(8)]
    assert max(ws) > 0.15
    assert ws[-1] < max(ws)
