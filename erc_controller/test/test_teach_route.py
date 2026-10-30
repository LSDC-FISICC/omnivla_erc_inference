"""teach_route.build_legs and checkpoint_controller_node.taught_leg: legs cut at the checkpoints, and the
rover joined to its taught leg at the nearest point. No ROS graph."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from erc_controller.teach_route import build_legs, local  # noqa: E402

LAT0, LON0 = 14.6095, -90.5058


def ll(e, n):
    return LAT0 + n / (math.radians(1) * 6378137.0), LON0 + e / (math.radians(1) * 6378137.0 * math.cos(math.radians(LAT0)))


def drive(points, step=0.3):
    """(t, lat, lon) along straight segments between points (east, north m), with a 0.2 m wobble."""
    out, t = [], 0.0
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        n = max(1, int(math.hypot(bx - ax, by - ay) / step))
        for i in range(n):
            f = i / n
            out.append((t, *ll(ax + f * (bx - ax) + 0.2 * math.sin(t), ay + f * (by - ay))))
            t += 0.5
    out.append((t, *ll(*points[-1])))
    return out


def test_legs_follow_the_detour_and_end_on_the_checkpoints():
    # A (0,0) -> around a wall end (0,10) -> B (8,10) -> C (8,0): the taught leg A->B has the corner
    track = drive([(0, 0), (0, 10), (8, 10), (8, 0)])
    cps = [ll(0, 0), ll(8, 10), ll(8, 0)]
    legs, notes = build_legs(track, cps)
    assert sorted(legs) == [1, 2, 3], notes
    corner = [local(LAT0, LON0, *p) for p in legs[2]]
    assert any(abs(e) < 0.6 and abs(n - 10) < 0.6 for e, n in corner)       # the detour is kept
    for k, cp in enumerate(cps, 1):
        e, n = local(LAT0, LON0, *legs[k][-1])
        ce, cn = local(LAT0, LON0, *cp)
        assert math.hypot(e - ce, n - cn) < 0.35
    assert len(legs[2]) < 12                                                 # simplified, not every fix


def test_unvisited_checkpoint_leaves_the_rest_to_astar():
    track = drive([(0, 0), (0, 10)])
    legs, notes = build_legs(track, [ll(0, 0), ll(0, 10), ll(30, 30), ll(0, 0)])
    assert sorted(legs) == [1, 2]
    assert any('checkpoint 3' in n for n in notes)


def test_taught_leg_joins_at_the_nearest_point():
    try:
        from erc_controller.checkpoint_controller_node import taught_leg
    except ImportError:          # needs the ROS message packages
        import pytest
        pytest.skip('checkpoint_controller_node needs a sourced workspace')
    pts = [ll(0, 0), ll(0, 5), ll(0, 10), ll(5, 10)]
    route, join = taught_leg(pts, *ll(0.5, 6))
    assert abs(join - math.hypot(0.5, 1)) < 0.05
    assert route[0] == ll(0.5, 6) and route[1] == pts[1] and route[-1] == pts[-1]
