"""local_planner.OdomPose (local_map_pose odom): the map's position by dead reckoning, the GPS only moving the
goal. No ROS."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from erc_controller import local_planner as lp  # noqa: E402


def test_steps_along_the_map_heading_whatever_the_odometry_yaw():
    d = lp.OdomPose()
    d.reset(1.0, 2.0)
    # the odometry drives 3 m along ITS yaw of 30 deg; the map's heading is north
    for k in range(31):
        s = 0.1 * k
        d.odom(s * math.cos(math.radians(30)), s * math.sin(math.radians(30)), math.radians(30), math.pi / 2)
    assert abs(d.x - 1.0) < 1e-9 and abs(d.y - 5.0) < 1e-9
    # backwards is negative
    for k in range(1, 11):
        s = 3.0 - 0.1 * k
        d.odom(s * math.cos(math.radians(30)), s * math.sin(math.radians(30)), math.radians(30), math.pi / 2)
    assert abs(d.y - 4.0) < 1e-9


def test_an_odometry_reset_is_not_a_step():
    d = lp.OdomPose()
    d.reset(0.0, 0.0)
    d.odom(5.0, 0.0, 0.0, 0.0)
    d.odom(0.0, 0.0, 0.0, 0.0)          # odometry restarted at its origin
    d.odom(0.1, 0.0, 0.0, 0.0)
    assert abs(d.x - 0.1) < 1e-9


def test_gps_moves_the_goal_slowly_never_the_position():
    d = lp.OdomPose(gps_tau_s=10.0)
    d.reset(0.0, 0.0)
    # the GPS sits 4 m east of where dead reckoning has the rover, with 1 m of noise
    rng = np.random.default_rng(0)
    for k in range(300):                                  # 30 s at 10 Hz
        d.gps(0.1 * k, 4.0 + rng.normal(0, 1.0), rng.normal(0, 1.0))
    assert d.x == 0.0 and d.y == 0.0
    assert abs(d.corr[0] - 4.0) < 0.6 and abs(d.corr[1]) < 0.6
    gx, gy = d.to_map(10.0, 0.0)                          # a checkpoint 10 m east by the GPS
    assert abs(gx - 6.0) < 0.6
    # one jump of 5 m moves the goal by 5 m x (1 - exp(-0.1/10)) only
    before = d.corr.copy()
    d.gps(30.0, before[0] + 5.0, before[1])
    assert np.hypot(*(d.corr - before)) < 0.06
