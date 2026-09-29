"""indoor_track: anchoring, the pseudo-fix, and the NYU route file.

The pseudo-fix is only right if omnivla_edge_node, running its own arithmetic
on it (UTM delta -> robot_frame_offset -> goal_bearing_from_offset), recovers
the goal's true offset in the track frame. That is tested end to end here.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erc_inference import indoor_track as it  # noqa: E402
from erc_inference import motion_control as mc  # noqa: E402

TRACK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'config', 'indoor_nyu_track.yaml')


def test_latlon_roundtrip():
    for x, y in [(0.0, 0.0), (36.2, 27.8), (-5.0, 12.0)]:
        bx, by = it.latlon_to_track(*it.track_to_latlon(x, y))
        assert bx == pytest.approx(x, abs=1e-6) and by == pytest.approx(y, abs=1e-6)


def test_compass_convention():
    assert it.yaw_to_compass_deg(0.0) == pytest.approx(90.0)             # +x is "east"
    assert it.yaw_to_compass_deg(math.pi / 2) == pytest.approx(0.0)      # +y is "north"
    assert it.yaw_to_compass_deg(math.pi) == pytest.approx(270.0)
    assert it.yaw_to_compass_deg(-math.pi / 2) == pytest.approx(180.0)


def test_anchor_maps_odom_onto_track():
    odom_at_start = it.Pose2D(3.0, -2.0, math.radians(40.0))
    start = it.Pose2D(0.0, 0.0, 0.0)
    T = it.anchor(start, odom_at_start)
    p = T.compose(odom_at_start)
    assert (p.x, p.y, p.yaw) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    # one meter forward in the rover's own heading is +x in the track
    fwd = odom_at_start.compose(it.Pose2D(1.0, 0.0, 0.0))
    q = T.compose(fwd)
    assert (q.x, q.y) == pytest.approx((1.0, 0.0), abs=1e-9)


@pytest.mark.parametrize('yaw_deg', [0.0, 90.0, 180.0, -135.0, 17.0])
@pytest.mark.parametrize('goal', [(1.5, 0.0), (0.0, 1.5), (-1.0, -1.0), (20.0, 3.0)])
def test_edge_node_arithmetic_recovers_track_offset(yaw_deg, goal):
    utm = pytest.importorskip('utm')
    rover = it.Pose2D(21.4, 5.0, math.radians(yaw_deg))
    gx, gy = rover.x + goal[0], rover.y + goal[1]
    cur = utm.from_latlon(*it.track_to_latlon(rover.x, rover.y))
    gl = utm.from_latlon(*it.track_to_latlon(gx, gy))
    rel_x, rel_y = mc.robot_frame_offset(gl[0] - cur[0], gl[1] - cur[1],
                                         it.yaw_to_compass_deg(rover.yaw))
    bearing = mc.goal_bearing_from_offset(rel_x, rel_y)
    expected = it.wrap(math.atan2(goal[1], goal[0]) - rover.yaw)       # left-positive
    assert bearing == pytest.approx(expected, abs=math.radians(0.05))
    assert math.hypot(rel_x, rel_y) == pytest.approx(math.hypot(*goal), rel=1e-3)


def test_nyu_track_file():
    track = it.load_track(TRACK_FILE)
    assert [c.sequence for c in track.checkpoints] == [1, 2, 3, 4, 5]
    assert [c.cone for c in track.checkpoints] == ['red', 'blue', 'green', 'yellow', 'orange']
    pts = [(track.start.x, track.start.y)]
    for c in track.checkpoints:
        pts += list(c.via) + [(c.x, c.y)]
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    # measured 128.4 m on the map against its stated 132 m
    assert 120.0 < length < 136.0
    # every leg is axis-aligned corridor driving: no diagonal segments
    for a, b in zip(pts, pts[1:]):
        assert min(abs(b[0] - a[0]), abs(b[1] - a[1])) < 0.5


def test_departure_pose_faces_next_leg():
    track = it.load_track(TRACK_FILE)
    assert track.departure_pose(0) == track.start
    p = track.departure_pose(1)                 # on CP1, facing the SE corner: +x
    assert (p.x, p.y) == (21.4, -0.4) and abs(p.yaw) < math.radians(2)
    p = track.departure_pose(2)                 # on CP2, facing north
    assert p.yaw == pytest.approx(math.pi / 2, abs=math.radians(1))
    with pytest.raises(ValueError):
        track.departure_pose(9)
