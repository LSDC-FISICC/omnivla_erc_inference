"""Rebuild a field bag's local map from its own /erc/free_space profiles, the rover placed by the GPS
(/erc/gps/filtered, as it ran before local_map_pose odom) and by dead reckoning (local_planner.OdomPose on
/erc/odometry/local along /erc/heading_deg) -- same LocalReplanner parameters as the field (goal mode,
corridor rule). Prints how much of the rover's own path the map puts inside a wall, saves a side-by-side figure.

    source /opt/ros/jazzy/setup.bash
    python3 map_pose_replay.py ~/lsdc/erc-omni-vla/rosbags/mission_galileo_30oct_b [--out fig.png]

mission_galileo_30oct_b (30-oct, the ramp walkway): path inside a wall 14.3 % (GPS) vs 1.9 % (odometry);
mission_galileo_goal (the planter plaza): 1.4 % vs 0.7 %.
"""
import argparse
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'erc_controller'))
from erc_controller import local_planner as lp  # noqa: E402

K = math.radians(1.0) * 6378137.0
LP = dict(goal_mode=True, keep_map=True, rejoin_respects_unseen=True, unseen_extend_m=0.1)


def read_bag(bag):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    topics = ['/erc/free_space', '/erc/heading_deg', '/erc/gps/filtered', '/erc/odometry/local', '/rosout']
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    cls = {t: get_message(types[t]) for t in topics}
    d = dict(scans=[], heading=[], gps=[], odom=[], starts=[], goals=[])
    while r.has_next():
        t, raw, ts = r.read_next()
        ts *= 1e-9
        m = deserialize_message(raw, cls[t])
        if t == '/erc/free_space':
            d['scans'].append((ts, m.angle_min, m.angle_increment, m.range_max, np.array(m.ranges, float)))
        elif t == '/erc/heading_deg':
            d['heading'].append((ts, m.data))
        elif t == '/erc/gps/filtered':
            d['gps'].append((ts, m.latitude, m.longitude))
        elif t == '/erc/odometry/local':
            q = m.pose.pose.orientation
            d['odom'].append((ts, m.pose.pose.position.x, m.pose.pose.position.y,
                              math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))))
        elif m.name == 'checkpoint_controller_node':
            g = re.search(r'Navigating to checkpoint sequence \d+ \(([-\d.]+), ([-\d.]+)\)', m.msg)
            if g:
                d['goals'].append((ts, float(g.group(1)), float(g.group(2))))
            g = re.search(r'leg frame origin \(([-\d.]+), ([-\d.]+)\)', m.msg)
            if g:
                d['starts'].append((ts, float(g.group(1)), float(g.group(2))))
    for k in ('heading', 'gps', 'odom', 'starts', 'goals'):
        d[k] = np.array(d[k], float)
    return d


def replay(d, pose, tau=20.0):
    """-> (LocalReplanner, path (t, x, y) in the map)."""
    t0, lat0, lon0 = d['starts'][0]
    kx = K * math.cos(math.radians(lat0))

    def xy(lat, lon):
        return (lon - lon0) * kx, (lat - lat0) * K
    H = d['heading']
    hu = np.degrees(np.unwrap(np.radians(H[:, 1])))

    def yaw(t):
        return math.radians(90.0 - float(np.interp(t, H[:, 0], hu)))
    ev = sorted([(r[0], 'g', i) for i, r in enumerate(d['gps']) if r[0] >= t0 - 2]
                + [(r[0], 'o', i) for i, r in enumerate(d['odom']) if r[0] >= t0 - 2]
                + [(s[0], 's', i) for i, s in enumerate(d['scans']) if s[0] >= t0])
    rp = lp.LocalReplanner(np.array([xy(g[1], g[2]) for g in d['goals']] + [(0.0, 0.0)]), **LP)
    dr, g, path = lp.OdomPose(tau), None, []
    for t, k, i in ev:
        if k == 'g':
            g = xy(*d['gps'][i, 1:3])
            if pose == 'odom':
                if not dr.ready:
                    dr.reset(*g)
                dr.gps(t, *g)
        elif k == 'o':
            if pose == 'odom':
                dr.odom(*d['odom'][i, 1:4], yaw(t))
        elif g is not None:
            x, y = (dr.x, dr.y) if pose == 'odom' and dr.ready else g
            _ts, a0, inc, rmax, r = d['scans'][i]
            b = np.degrees(a0 + np.arange(len(r)) * inc)
            hit = np.isfinite(r) & (r < rmax - 0.05)
            rp.observe(t, x, y, yaw(t), b, np.where(np.isfinite(r), r, rmax), hit)
            path.append((t, x, y))
    return rp, np.array(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    d = read_bag(a.bag)
    if not len(d['starts']):
        sys.exit('no "leg frame origin" in /rosout: local_replan was off or the bag misses the leg start')
    res = {pose: replay(d, pose) for pose in ('gps', 'odom')}
    for pose, (rp, P) in res.items():
        c = rp.map.clearance_at(P[:, 1], P[:, 2])
        print(f'{pose:4s}: path {np.hypot(*np.diff(P[:, 1:], axis=0).T).sum():5.1f} m, inside a wall '
              f'(<= 0.1 m) {100 * np.mean(c <= 0.1):4.1f} %, within inflation {100 * np.mean(c <= 0.45):4.1f} %')
    if a.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        for axx, (pose, (rp, P)) in zip(ax, res.items()):
            m = rp.map
            img = np.where(m.seen, 1 - 1 / (1 + np.exp(-m.L)), 0.55)
            axx.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=1,
                       extent=[m.ox, m.ox + m.w * m.p['resolution_m'], m.oy, m.oy + m.h * m.p['resolution_m']])
            axx.plot(P[:, 1], P[:, 2], 'c-', lw=1)
            axx.set_xlim(P[:, 1].min() - 6, P[:, 1].max() + 6)
            axx.set_ylim(P[:, 2].min() - 6, P[:, 2].max() + 6)
            axx.set_aspect('equal')
            axx.set_title(f'{os.path.basename(a.bag.rstrip("/"))}: rover placed by {pose}')
        plt.tight_layout()
        plt.savefig(a.out, dpi=80)


if __name__ == '__main__':
    main()
