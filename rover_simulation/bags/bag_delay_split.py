"""How much of the ~1.3 s command -> motion delay is actuation, and how much is telemetry age?

Mission16sept.md 7.2 left two readings open, because the gyro alone cannot tell them apart:
  1. real delay: telemetry arrives ~1.16 s late, actuation is ~0.2 s
  2. rover clock ~1.1 s behind: telemetry is fresh, actuation is ~1.3 s
and suggested the way out: a second, independent clock. The front camera is stamped
by the SDK server (image_latency ~0.04 s) and the IMU by the rover. So the yaw rate is
measured twice and cross-correlated against the command, which is recorded locally
(its bag time IS when it was sent):

  command (bag time)  ->  camera rotation, server stamp    = actuation + camera capture
  command (bag time)  ->  gyro, rover stamp                = actuation (+ rover clock error)
  command (bag time)  ->  gyro, bag time (arrival)         = the end-to-end delay the
                                                             controller lives with
  camera stamp vs gyro stamp                               = rover clock error

Camera rotation: horizontal phase correlation between consecutive frames (grey,
downsampled), converted with the calibrated f; only meaningful for in-place turns,
which the side-step bags have plenty of.

    source /opt/ros/jazzy/setup.bash
    python3 bag_delay_split.py <bag> [<bag> ...] [--plot out.png]
"""
import argparse
import math

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

F_PX = 512.3          # full-res focal length, TAREA1 2.3
SCALE = 0.25          # frames are phase-correlated at 1/4 resolution
RATE = 20.0           # Hz, common resampling grid


def read(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=''), rosbag2_py.ConverterOptions('', ''))
    r.set_filter(rosbag2_py.StorageFilter(topics=['/cmd_vel', '/erc/imu', '/erc/front_camera']))
    ty = {t.name: t.type for t in r.get_all_topics_and_types()}
    cmd, gyro, vis = [], [], []
    prev, prev_stamp = None, None
    win = None
    while r.has_next():
        tp, d, t = r.read_next()
        t *= 1e-9
        m = deserialize_message(d, get_message(ty[tp]))
        if tp == '/cmd_vel':
            cmd.append((t, m.angular.z, m.linear.x))
        elif tp == '/erc/imu':
            st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            gyro.append((st, t, m.angular_velocity.z))
        else:
            st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if prev_stamp is not None and st <= prev_stamp + 1e-6:
                continue          # repeated frame
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
            g = cv2.cvtColor(cv2.resize(img, None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY).astype(np.float32)
            g = g[: int(g.shape[0] * 0.6)]       # above the near ground: pure rotation shifts it uniformly
            if win is None:
                win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
            if prev is not None:
                (dx, _dy), resp = cv2.phaseCorrelate(prev, g, win)
                dt = st - prev_stamp
                if 0.02 < dt < 0.5 and resp > 0.1:
                    # image content moves right when the rover turns left (+yaw)
                    vis.append((st, t, math.atan(dx / (F_PX * SCALE)) / dt))
            prev, prev_stamp = g, st
    return np.array(cmd), np.array(gyro), np.array(vis)


def series(t, v, grid):
    order = np.argsort(t)
    return np.interp(grid, t[order], v[order], left=np.nan, right=np.nan)


def lag_of(grid, a, b, max_lag=3.0):
    """Delay (s) of b behind a, by normalised cross-correlation over valid samples."""
    best, best_c = None, -2.0
    n = int(max_lag * RATE)
    for k in range(-n // 3, n + 1):
        if k >= 0:
            x, y = a[: len(a) - k], b[k:]
        else:
            x, y = a[-k:], b[: len(b) + k]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 50:
            continue
        xx, yy = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
        c = float(xx @ yy / (np.linalg.norm(xx) * np.linalg.norm(yy) + 1e-12))
        if c > best_c:
            best, best_c = k / RATE, c
    return best, best_c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--plot')
    a = ap.parse_args()
    rows = []
    for bag in a.bags:
        cmd, gyro, vis = read(bag)
        t0, t1 = cmd[0, 0], cmd[-1, 0]
        grid = np.arange(t0, t1, 1 / RATE)
        w_cmd = series(cmd[:, 0], cmd[:, 1], grid)
        # yaw-rate sign of the visual estimate is fixed by correlation with the gyro below
        w_vis_srv = series(vis[:, 0], vis[:, 2], grid)
        w_gyr_rov = series(gyro[:, 0], gyro[:, 2], grid)
        w_gyr_bag = series(gyro[:, 1], gyro[:, 2], grid)
        sign = np.sign(np.nansum(w_vis_srv * w_gyr_rov)) or 1.0
        w_vis_srv *= sign
        L = {
            'cmd -> camera (server stamp)': lag_of(grid, w_cmd, w_vis_srv),
            'cmd -> gyro (rover stamp)': lag_of(grid, w_cmd, w_gyr_rov),
            'cmd -> gyro (arrival, bag time)': lag_of(grid, w_cmd, w_gyr_bag),
            'camera (server) -> gyro (rover stamp)': lag_of(grid, w_vis_srv, w_gyr_rov),
            'gyro rover stamp -> gyro arrival': lag_of(grid, w_gyr_rov, w_gyr_bag),
        }
        age = np.median(gyro[:, 1] - gyro[:, 0])
        cam_age = np.median(vis[:, 1] - vis[:, 0])
        print(f'== {bag.rstrip("/").split("/")[-1]}  ({len(vis)} camera rotations, {len(gyro)} gyro samples)')
        print(f'   gyro age on arrival (bag - stamp): p50 {age:.2f} s; camera frame age: p50 {cam_age:.2f} s')
        for k, (lag, c) in L.items():
            print(f'   {k:40s} lag {lag:+.2f} s   (corr {c:.2f})')
        rows.append(L)
        if a.plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(13, 4))
            tt = grid - t0
            ax.plot(tt, w_cmd, 'k-', lw=1, label='command (sent)')
            ax.plot(tt, w_vis_srv, 'c-', lw=1, label='camera rotation (server stamp)')
            ax.plot(tt, w_gyr_rov, 'm-', lw=1, alpha=.7, label='gyro (rover stamp)')
            ax.plot(tt, w_gyr_bag, 'r-', lw=1, alpha=.5, label='gyro (arrival)')
            ax.set_xlim(*(a_ for a_ in (tt[np.nanargmax(np.abs(w_cmd))] - 15, tt[np.nanargmax(np.abs(w_cmd))] + 25)))
            ax.set_ylabel('yaw rate (rad/s)')
            ax.set_xlabel('s')
            ax.legend(fontsize=7)
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(a.plot.replace('.png', f'_{len(rows)}.png'), dpi=90)


if __name__ == '__main__':
    main()
