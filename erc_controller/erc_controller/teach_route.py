"""Teach and repeat: the route of each leg from a drive done by hand, instead of OSM's A*.

Why. OSM does not have the walls, planters and stairs of a campus walkway; A* then draws legs
straight through them, and the local map only sees +-40 deg and ~3 m ahead -- in a 2 m brick
walkway it could not find the way round (Galileo plaza, 29-oct). A drive by hand along the
mission says where the rover CAN go. checkpoint_controller_node (taught_route_file) drives
each leg along it; the local map and the drive still deal with what is new (people, tarps).

    # 1. record while you drive the mission by hand, checkpoint after checkpoint (SDK only)
    ros2 run erc_controller teach_route record --out ~/lsdc/erc-omni-vla/missions/galileo-robot.track.csv
    # 2. cut it into legs at the checkpoints (also from a bag: --track <bag dir>, /erc/gps/filtered)
    ros2 run erc_controller teach_route build --mission galileo-robot \\
        --track ~/lsdc/erc-omni-vla/missions/galileo-robot.track.csv
    #    -> missions/galileo-robot.route.yaml, checked: every leg found, its length, gaps
    # 3. ros2 run erc_controller checkpoint_controller_node --ros-args \\
    #        -p taught_route_file:=$HOME/lsdc/erc-omni-vla/missions/galileo-robot.route.yaml ...

How a leg is cut. The track is walked in time; checkpoint k is "visited" at the closest
approach of the first pass within --radius-m of it after checkpoint k-1's visit. Leg k is the
track between the two visits. Leg 1 starts where the track starts. A checkpoint never visited
leaves its leg (and the following ones) out: the controller then plans those with A* as before.
The polyline is simplified (Douglas-Peucker, --simplify-m) so the carrot follows the shape,
not the GPS jitter.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request

import yaml

MISSIONS_DIR = os.path.expanduser('~/lsdc/erc-omni-vla/missions')
K = math.radians(1.0) * 6378137.0


def local(lat0, lon0, lat, lon):
    return (lon - lon0) * K * math.cos(math.radians(lat0)), (lat - lat0) * K


def simplify(pts, eps):
    """Douglas-Peucker on [(e, n, i)]: keeps the end points."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    best, idx = -1.0, 0
    for i, p in enumerate(pts[1:-1], 1):
        d = (abs(dy * (p[0] - a[0]) - dx * (p[1] - a[1])) / L) if L > 1e-9 else math.hypot(p[0] - a[0], p[1] - a[1])
        if d > best:
            best, idx = d, i
    if best <= eps:
        return [a, b]
    return simplify(pts[:idx + 1], eps)[:-1] + simplify(pts[idx:], eps)


def load_mission(mission):
    path = mission if mission.endswith('.yaml') else os.path.join(MISSIONS_DIR, mission + '.yaml')
    doc = yaml.safe_load(open(path))
    cps = []
    for c in doc['checkpoints']:
        cps.append((float(c[0]), float(c[1])) if isinstance(c, (list, tuple)) else (float(c['lat']), float(c['lon'])))
    return path, cps


def load_track(track):
    """[(t, lat, lon)] from a CSV of `record` or a rosbag2 directory (/erc/gps/filtered, else /erc/gps)."""
    if os.path.isdir(track):
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import NavSatFix
        r = rosbag2_py.SequentialReader()
        storage = 'mcap' if any(f.endswith('.mcap') for f in os.listdir(track)) else 'sqlite3'
        r.open(rosbag2_py.StorageOptions(uri=track, storage_id=storage), rosbag2_py.ConverterOptions('', ''))
        topics = {t.name for t in r.get_all_topics_and_types()}
        topic = '/erc/gps/filtered' if '/erc/gps/filtered' in topics else '/erc/gps'
        r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
        out = []
        while r.has_next():
            _, raw, ts = r.read_next()
            m = deserialize_message(raw, NavSatFix)
            if abs(m.latitude) > 1e-6:
                out.append((ts * 1e-9, m.latitude, m.longitude))
        return out, topic
    out = []
    with open(track) as f:
        for row in csv.reader(f):
            if row and not row[0].startswith('#'):
                out.append((float(row[0]), float(row[1]), float(row[2])))
    return out, 'csv'


def build_legs(track, cps, radius_m=3.0, simplify_m=0.25):
    """-> (legs {sequence: [(lat, lon), ...]}, report lines)."""
    lat0, lon0 = cps[0]
    tr = [(t, *local(lat0, lon0, la, lo), la, lo) for t, la, lo in track]
    legs, notes, start = {}, [], 0
    for k, (cla, clo) in enumerate(cps):
        ce, cn = local(lat0, lon0, cla, clo)
        d = [math.hypot(p[1] - ce, p[2] - cn) for p in tr]
        first = next((i for i in range(start, len(tr)) if d[i] <= radius_m), None)
        if first is None:
            notes.append(f'checkpoint {k + 1}: never within {radius_m:g} m after checkpoint {k}: '
                         f'legs {k + 1}..{len(cps)} left to A*')
            break
        end = first                                   # closest approach of this pass
        while end + 1 < len(tr) and d[end + 1] <= radius_m:
            end += 1
        visit = min(range(first, end + 1), key=lambda i: d[i])
        seg = [(p[1], p[2], i) for i, p in enumerate(tr[start:visit + 1], start)]
        kept = simplify(seg, simplify_m) if len(seg) > 1 else seg
        pts = [(tr[i][3], tr[i][4]) for _, _, i in kept]
        if math.hypot(tr[visit][1] - ce, tr[visit][2] - cn) > 0.3:
            pts.append((cla, clo))                    # the leg ends ON the checkpoint
        length = sum(math.hypot(*[a - b for a, b in zip(local(lat0, lon0, *p), local(lat0, lon0, *q))])
                     for p, q in zip(pts, pts[1:]))
        legs[k + 1] = pts
        notes.append(f'leg {k + 1}: {len(seg)} fixes -> {len(pts)} points, {length:.1f} m, '
                     f'closest {d[visit]:.1f} m to checkpoint {k + 1}')
        start = visit
    return legs, notes


def cmd_build(a):
    mpath, cps = load_mission(a.mission)
    track, topic = load_track(a.track)
    if len(track) < 2:
        sys.exit(f'{a.track}: no GPS in it')
    legs, notes = build_legs(track, cps, a.radius_m, a.simplify_m)
    for n in notes:
        print(n)
    out = a.out or mpath[:-len('.yaml')] + '.route.yaml'
    doc = {'mission': os.path.basename(mpath), 'track': os.path.abspath(a.track), 'topic': topic,
           'radius_m': a.radius_m, 'simplify_m': a.simplify_m,
           'legs': {int(k): [[round(la, 8), round(lo, 8)] for la, lo in v] for k, v in legs.items()}}
    with open(out, 'w') as f:
        f.write('# taught route: one polyline per leg (the leg ENDING at checkpoint <sequence>), from a drive by\n'
                '# hand; teach_route.py build. checkpoint_controller_node taught_route_file reads it.\n')
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
    print(f'-> {out} ({len(legs)}/{len(cps)} legs)')
    return 0 if len(legs) == len(cps) else 1


def cmd_record(a):
    print(f'recording {a.sdk}/data at {a.rate_hz:g} Hz into {a.out} -- drive the mission by hand, Ctrl-C to stop')
    n = 0
    with open(a.out, 'a') as f:
        f.write(f'# t, lat, lon, fix_quality -- teach_route record {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        try:
            while True:
                try:
                    d = json.load(urllib.request.urlopen(a.sdk + '/data', timeout=3))
                    la, lo = d.get('latitude'), d.get('longitude')
                    if la and lo and abs(float(la)) > 1e-6:
                        f.write(f'{time.time():.3f},{float(la):.8f},{float(lo):.8f},{d.get("fix_quality")}\n')
                        f.flush()
                        n += 1
                        if n % 20 == 0:
                            print(f'  {n} fixes, last {float(la):.7f}, {float(lo):.7f}', flush=True)
                except Exception as exc:          # SDK busy: keep going
                    print(f'  (SDK: {exc})', flush=True)
                time.sleep(1.0 / a.rate_hz)
        except KeyboardInterrupt:
            print(f'\n{n} fixes in {a.out}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record')
    r.add_argument('--out', required=True)
    r.add_argument('--sdk', default='http://localhost:8000')
    r.add_argument('--rate-hz', type=float, default=2.0)
    b = sub.add_parser('build')
    b.add_argument('--mission', required=True, help='slug in ~/lsdc/erc-omni-vla/missions or a path')
    b.add_argument('--track', required=True, help='CSV from record, or a rosbag2 directory')
    b.add_argument('--out', default='')
    b.add_argument('--radius-m', type=float, default=3.0)
    b.add_argument('--simplify-m', type=float, default=0.25)
    a = ap.parse_args(argv)
    return cmd_build(a) if a.cmd == 'build' else cmd_record(a)


if __name__ == '__main__':
    sys.exit(main())
