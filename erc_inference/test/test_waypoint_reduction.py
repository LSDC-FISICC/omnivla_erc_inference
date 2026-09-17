"""
Regression suite for reducing a planned A* path to drivable GPS waypoints.

This exists because of a real bug. checkpoint_controller_node._path_to_waypoints
used to keep exactly 10 evenly-indexed poses regardless of path length, and the
controller drives straight lines between consecutive waypoints. Replaying that
indexing over real A* output put the rover straight back through the buildings
the planner had just routed around -- 27 to 47 lethal cells crossed across three
scenarios. At the default 0.15 m/cell a 500 m leg is ~3300 path cells, so the
waypoints landed ~50 m apart and every finer detour became a straight line
nobody had checked.

test_old_fixed_count_sampling_would_cross_buildings below keeps that failure
reproducible, so nobody reintroduces it.

Depends on erc_static_map for A* itself: generating the paths with the real
planner rather than hand-drawn polylines is the point -- the shapes A* actually
produces are what has to survive the reduction.
"""

import importlib.util
import math
import os

import erc_static_map.erc_astar_planner_node as planner

from geometry_msgs.msg import Pose

from nav_msgs.msg import OccupancyGrid

import numpy as np

import pytest

RESOLUTION = 1.0
MAX_SPACING_M = 15.0


def _load_controller():
    """Import the node module by path: erc_inference is not installed as a package."""
    # in the test environment, and importing it for two static methods should not
    # require one.
    path = os.path.join(os.path.dirname(__file__), '..', 'erc_inference',
                        'checkpoint_controller_node.py')
    spec = importlib.util.spec_from_file_location('checkpoint_controller_node', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CheckpointControllerNode


Controller = _load_controller()


def make_grid(occupied, resolution=RESOLUTION):
    grid = OccupancyGrid()
    grid.info.resolution = resolution
    grid.info.height, grid.info.width = occupied.shape
    pose = Pose()
    pose.position.x = pose.position.y = 0.0
    pose.orientation.w = 1.0
    grid.info.origin = pose
    grid.data = np.where(occupied, 100, 0).astype(np.int8).flatten().tolist()
    return grid


def plan_points(occupied, start, goal):
    path = planner.astar_grid(occupied, start, goal)
    assert path is not None, 'test scenario must be solvable'
    return [(float(c), float(r)) for c, r in path]


def lethal_crossings(occupied, waypoints):
    """How many samples along the straight lines between waypoints land on a."""
    # lethal cell. Sampled at a quarter cell, which cannot step over a cell.
    hits = 0
    for a, b in zip(waypoints, waypoints[1:]):
        steps = max(2, int(4 * (abs(b[0] - a[0]) + abs(b[1] - a[1]))))
        for i in range(steps + 1):
            t = i / steps
            c = int(a[0] + (b[0] - a[0]) * t)
            r = int(a[1] + (b[1] - a[1]) * t)
            if 0 <= r < occupied.shape[0] and 0 <= c < occupied.shape[1] and occupied[r, c]:
                hits += 1
    return hits


def old_fixed_count_sampling(points):
    """Reproduce the behaviour this suite exists to prevent coming back."""
    return [points[0]] + [points[min(int(len(points) * i / 10.0), len(points) - 1)]
                          for i in range(1, 11)]


# ------------------------------------------------------------- scenarios

def single_building():
    occupied = np.zeros((400, 400), dtype=bool)
    occupied[150:250, 180:220] = True
    return occupied, (20, 200), (380, 200)


def corridor_with_a_doorway():
    occupied = np.zeros((400, 400), dtype=bool)
    occupied[100:300, 150:170] = True
    occupied[100:300, 230:250] = True
    occupied[100:120, 150:250] = True
    return occupied, (200, 50), (200, 380)


def scattered_buildings():
    occupied = np.zeros((400, 400), dtype=bool)
    rng = np.random.default_rng(3)
    for _ in range(12):
        r0, c0 = rng.integers(60, 340, 2)
        occupied[r0:r0 + 35, c0:c0 + 35] = True
    occupied[200, 20] = occupied[200, 380] = False
    return occupied, (20, 200), (380, 200)


SCENARIOS = [single_building, corridor_with_a_doorway, scattered_buildings]


@pytest.mark.parametrize('scenario', SCENARIOS, ids=lambda f: f.__name__)
def test_no_segment_crosses_a_building(scenario):
    occupied, start, goal = scenario()
    points = plan_points(occupied, start, goal)
    kept = Controller._shortcut_path(points, make_grid(occupied), MAX_SPACING_M)
    assert lethal_crossings(occupied, kept) == 0


@pytest.mark.parametrize('scenario', SCENARIOS, ids=lambda f: f.__name__)
def test_old_fixed_count_sampling_would_cross_buildings(scenario):
    """The bug, kept reproducible. If this ever stops crossing buildings the."""
    # scenario has gone slack and stopped testing anything.
    occupied, start, goal = scenario()
    points = plan_points(occupied, start, goal)
    assert lethal_crossings(occupied, old_fixed_count_sampling(points)) > 0


@pytest.mark.parametrize('scenario', SCENARIOS, ids=lambda f: f.__name__)
def test_endpoints_are_preserved(scenario):
    occupied, start, goal = scenario()
    points = plan_points(occupied, start, goal)
    kept = Controller._shortcut_path(points, make_grid(occupied), MAX_SPACING_M)
    assert kept[0] == points[0]
    assert kept[-1] == points[-1]


@pytest.mark.parametrize('scenario', SCENARIOS, ids=lambda f: f.__name__)
def test_no_segment_exceeds_the_spacing_cap(scenario):
    occupied, start, goal = scenario()
    points = plan_points(occupied, start, goal)
    kept = Controller._shortcut_path(points, make_grid(occupied), MAX_SPACING_M)
    assert max(math.dist(a, b) for a, b in zip(kept, kept[1:])) <= MAX_SPACING_M + 1e-9


def test_open_ground_collapses_to_few_waypoints():
    """Shortcutting has to actually shortcut, or the controller gets a goal per cell."""
    occupied = np.zeros((100, 100), dtype=bool)
    points = plan_points(occupied, (10, 10), (90, 90))
    kept = Controller._shortcut_path(points, make_grid(occupied), MAX_SPACING_M)
    assert len(kept) < len(points) / 5


def test_degenerate_inputs_do_not_raise():
    grid = make_grid(np.zeros((10, 10), dtype=bool))
    assert Controller._shortcut_path([], grid, MAX_SPACING_M) == []
    assert Controller._shortcut_path([(0.0, 0.0)], grid, MAX_SPACING_M) == [(0.0, 0.0)]


def test_unusable_grid_returns_the_path_untouched():
    """Falling back to a fixed count here is what caused the original bug, so an."""
    # unusable costmap must return everything rather than a truncation.
    points = plan_points(np.zeros((100, 100), dtype=bool), (10, 10), (90, 90))
    broken = OccupancyGrid()
    broken.info.resolution = 0.0
    assert Controller._shortcut_path(points, broken, MAX_SPACING_M) == points


def test_distance_resampling_fallback_scales_with_leg_length():
    """The no-costmap fallback samples by distance, so the interval does not grow."""
    # with the leg the way a fixed count does.
    points = plan_points(np.zeros((100, 100), dtype=bool), (10, 10), (90, 90))
    coarse = Controller._resample_by_distance(points, 15.0)
    fine = Controller._resample_by_distance(points, 5.0)
    assert len(fine) > len(coarse)
    assert coarse[0] == points[0] and coarse[-1] == points[-1]


def test_segment_check_treats_off_grid_as_blocked():
    """Off-map is exactly the terrain nothing has checked."""
    occupied = np.zeros((10, 10), dtype=bool)
    occ, res, ox, oy = Controller._grid_occupancy(make_grid(occupied))
    assert Controller._segment_is_clear(occ, res, ox, oy, (1.0, 1.0), (8.0, 8.0))
    assert not Controller._segment_is_clear(occ, res, ox, oy, (1.0, 1.0), (25.0, 1.0))
