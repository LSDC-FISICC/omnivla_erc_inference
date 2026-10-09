"""Fake floor for an end-to-end test of the potted-plant search (run_e2e_plants.sh).

The REAL image_checkpoint_controller_node (started by plants_controller.launch.py, so with
explore:=true) and carrot_controller_node run against:
  - the plant of fake_world.py (1.3 s command delay, k_v 1.11, k_w PLANT_KW_MOVING / 1.18 in
    place), integrated on the wall clock, with dead-reckoning errors on the published odometry
    and an odom origin offset from the mission frame;
  - the floor of test/explore_search_sim.py, which the mission knows nothing about: two 2 m
    corridors in an L, a 5 x 4 m room, four potted plants (discs: obstacles too);
  - /erc/free_space from test/obstacles.profile and /erc/objects as object_detector_node
    publishes it (<= 8 m, +-50 deg, p 0.8, line of sight, 1.5 deg noise, range 8 % short and 8 % noisy), both from
    where the rover was IMAGE_LAG_S ago.
It also checks what the node publishes to look at: /erc/explore/map and /erc/explore/objects.

    python3 fake_plants_world.py <seconds> <seed> <plants to show>
"""
import collections
import json
import math
import os
import re
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import Log
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'erc_inference', 'test'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'erc_inference'))
import explore_search_sim as ess  # noqa: E402
import obstacles as obs  # noqa: E402

PLANT_KW_MOVING = float(os.environ.get('PLANT_KW_MOVING', 1.1))
PLANT_KW_IN_PLACE = float(os.environ.get('PLANT_KW_IN_PLACE', 1.18))
WHEEL_SCALE = float(os.environ.get('WHEEL_SCALE', 0.01))
GYRO_SCALE = float(os.environ.get('GYRO_SCALE', 0.01))
GYRO_BIAS_DEG_MIN = float(os.environ.get('GYRO_BIAS_DEG_MIN', 0.5))
IMAGE_LAG_S = float(os.environ.get('IMAGE_LAG_S', 0.8))
ODOM_OFFSET = (5.0, -3.0, math.radians(30.0))
STATE = {'done': False, 'shown': []}


class World(Node):
    def __init__(self, seed, n_goals):
        super().__init__('fake_plants_world')
        self.rng = np.random.default_rng(seed)
        self.n_goals = n_goals
        self.walls = ess.world()
        self.plants = ess.PLANTS
        self.world_all = self.walls + [obs.Disc(tuple(p), ess.PLANT_R) for p in self.plants]
        self.x = self.y = self.th = 0.0
        self.ex = self.ey = self.eth = 0.0
        self.v = self.w = 0.0
        self.cmds = collections.deque([(-1e9, 0.0, 0.0)])
        self.hist = collections.deque()
        self.t0 = self._now()
        self.last = None
        self.hits, self.hit_now, self.min_clear = [], False, 9.0
        self.pose_err = []
        self.est_pose = None
        self.maps, self.map_last, self.markers_last = 0, None, None
        self.going_to = (0.0, 0.0)
        self.odom = self.create_publisher(Odometry, '/erc/odometry/local', 10)
        self.scan = self.create_publisher(LaserScan, '/erc/free_space', 10)
        self.obj_pub = self.create_publisher(String, '/erc/objects', 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.create_subscription(PoseStamped, '/erc/indoor/pose', self._pose, 10)
        self.create_subscription(Log, '/rosout', self._log, 50)
        self.create_subscription(OccupancyGrid, '/erc/explore/map', self._map, 1)
        self.create_subscription(MarkerArray, '/erc/explore/objects', self._markers, 1)
        self.create_timer(0.05, self._step)
        self.create_timer(1 / 30, self._pub_odom)
        self.create_timer(1 / 3, self._camera)
        print('plants: ' + ', '.join(f'({p[0]:.1f},{p[1]:.1f})' for p in self.plants), flush=True)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def t(self):
        return self._now() - self.t0

    def _cmd(self, m):
        self.cmds.append((self._now(), m.linear.x, m.angular.z))

    def _pose(self, m):
        self.est_pose = (m.pose.position.x, m.pose.position.y)

    def _map(self, m):
        self.maps += 1
        self.map_last = m

    def _markers(self, m):
        self.markers_last = m

    def _log(self, m):
        if m.name != 'image_checkpoint_controller_node':
            return
        if any(k in m.msg for k in ('object ', 'shown', 'Mission', 'goal ', 'no path', 'explore', 'frontier',
                                    'stuck', 'Anchored', 'ready')):
            print(f'  {self.t():6.1f}s  {m.msg[:150]}', flush=True)
        got = re.search(r'object \d+/\d+: .* at \((-?[\d.]+),(-?[\d.]+)\)', m.msg)
        if got:                      # where the mission thinks the plant it is going to stands
            self.going_to = (float(got.group(1)), float(got.group(2)))
        if 'shown.' in m.msg:        # "cone n/N shown.": which plant that was, and how far it really is
            which = int(np.hypot(*(self.plants - self.going_to).T).argmin())
            d = float(np.hypot(*(self.plants[which] - (self.x, self.y))))
            STATE['shown'].append((round(self.t()), which, round(d, 2)))
            STATE['done'] = len(STATE['shown']) >= self.n_goals
        if 'nothing left to explore' in m.msg or 'Mission failed' in m.msg:
            STATE['done'] = True

    def _step(self):
        now = self._now()
        dt = 0.05 if self.last is None else min(0.2, now - self.last)
        self.last = now
        while len(self.cmds) > 1 and self.cmds[1][0] <= now - 1.3:
            self.cmds.popleft()
        _, lin, ang = self.cmds[0]
        tv = 1.11 * lin if abs(lin) >= 0.15 else 0.0
        moving = abs(self.v) > 0.03 or tv != 0.0
        tw = (PLANT_KW_MOVING * ang) if moving else (PLANT_KW_IN_PLACE * ang if abs(ang) >= 0.15 else 0.0)
        self.v += (tv - self.v) * min(1, dt / 0.47)
        self.w += (tw - self.w) * min(1, dt / 0.35)
        self.th += self.w * dt
        nx, ny = self.x + self.v * math.cos(self.th) * dt, self.y + self.v * math.sin(self.th) * dt
        c = obs.clearance((nx, ny), self.world_all)
        h = c <= obs.FOOTPRINT_W / 2
        if h and not self.hit_now:
            self.hits.append((round(self.t(), 1), round(self.x, 1), round(self.y, 1)))
        self.hit_now = h
        moved = 0.0
        if not h or self.v < 0:       # a wall does not let it through (it can still back off)
            self.x, self.y = nx, ny
            moved = self.v
        self.min_clear = min(self.min_clear, c)
        self.eth += (self.w * (1 + GYRO_SCALE) + math.radians(GYRO_BIAS_DEG_MIN) / 60.0) * dt
        self.ex += moved * (1 + WHEEL_SCALE) * math.cos(self.eth) * dt
        self.ey += moved * (1 + WHEEL_SCALE) * math.sin(self.eth) * dt
        self.hist.append((now, self.x, self.y, self.th))
        while len(self.hist) > 1 and self.hist[1][0] <= now - IMAGE_LAG_S:
            self.hist.popleft()
        if self.est_pose is not None:
            self.pose_err.append(math.hypot(self.est_pose[0] - self.x, self.est_pose[1] - self.y))

    def _pub_odom(self):
        ox, oy, oth = ODOM_OFFSET
        c, s = math.cos(oth), math.sin(oth)
        o = Odometry()
        o.header.stamp = self.get_clock().now().to_msg()
        o.header.frame_id, o.child_frame_id = 'odom', 'base_link'
        o.pose.pose.position.x = ox + c * self.ex - s * self.ey
        o.pose.pose.position.y = oy + s * self.ex + c * self.ey
        yaw = oth + self.eth
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        o.twist.twist.linear.x, o.twist.twist.angular.z = float(self.v), float(self.w)
        self.odom.publish(o)

    def _camera(self):
        _, lx, ly, lth = self.hist[0] if self.hist else (0, self.x, self.y, self.th)
        b, free, caps = obs.profile(lx, ly, lth, self.world_all, self.rng)
        free = np.where(free < caps - 1e-3, free, 3.0)
        s = LaserScan()
        s.header.stamp = self.get_clock().now().to_msg()
        s.angle_min, s.angle_max = math.radians(b[0]), math.radians(b[-1])
        s.angle_increment = math.radians(b[1] - b[0])
        s.range_max = 3.0
        s.ranges = [float(x) for x in free]
        self.scan.publish(s)
        o = np.array([lx, ly])
        dets = []
        for p in self.plants:
            d = p - o
            r = float(np.hypot(*d))
            brg = math.degrees(math.remainder(math.atan2(d[1], d[0]) - lth, 2 * math.pi))
            if abs(brg) > 50.0 or not 0.4 < r < 8.0 or self.rng.random() > 0.8:
                continue
            if min((w.ray_hit(o, d / r, r) for w in self.walls), default=math.inf) < r - 0.05:
                continue
            rng_m = r * (0.92 + self.rng.normal(0, 0.08))     # the depth model's range: short and noisy
            dets.append({'label': 'potted plant', 'conf': 0.8, 'bearing_deg': brg + self.rng.normal(0, 1.5),
                         'range_m': rng_m, 'range_base_m': rng_m, 'range_scan_m': None, 'clipped': False})
        self.obj_pub.publish(String(data=json.dumps({'stamp': self._now(), 'detections': dets})))


def main():
    dur, seed, n_goals = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rclpy.init()
    n = World(seed, n_goals)
    drive = len(sys.argv) > 4 and sys.argv[4] == 'drive'     # no mission: this drives, for local_map_node
    if drive:
        pub = n.create_publisher(Twist, '/cmd_vel', 10)
        n.create_subscription(OccupancyGrid, '/erc/local_map', n._map, 1)
        n.create_subscription(MarkerArray, '/erc/local_map/objects', n._markers, 1)
        n.create_timer(0.1, lambda: pub.publish(Twist(linear=Vector3(x=0.25)) if n.x < 10.5 else Twist()))
    while rclpy.ok() and n.t() < dur and not STATE['done']:
        rclpy.spin_once(n, timeout_sec=0.01)
    t_end = n.t()
    for _ in range(200):             # the last map
        rclpy.spin_once(n, timeout_sec=0.01)
    pe = np.array(n.pose_err) if n.pose_err else np.array([np.nan])
    shown = STATE['shown']
    print(f'plants_e2e seed={seed}: t={t_end:.0f}s shown={len(shown)}/{n_goals} distinct={len({s[1] for s in shown})} '
          f'(t, plant, true m)={shown} hits={len(n.hits)} {n.hits[:6]} min_clear={n.min_clear:.2f} '
          f'pose_err p50={np.nanmedian(pe):.2f} max={np.nanmax(pe):.2f}', flush=True)
    g = n.map_last
    if g is None:
        print('map: NOTHING on /erc/explore/map', flush=True)
    else:
        d = np.asarray(g.data, np.int16).reshape(g.info.height, g.info.width)
        # occupied cells back in the mission frame: they must lie on the walls and the plants
        rows, cols = np.nonzero(d > 80)
        yaw = 2 * math.atan2(g.info.origin.orientation.z, g.info.origin.orientation.w)
        cx, cy = (cols + 0.5) * g.info.resolution, (rows + 0.5) * g.info.resolution
        xo = g.info.origin.position.x + math.cos(yaw) * cx - math.sin(yaw) * cy
        yo = g.info.origin.position.y + math.sin(yaw) * cx + math.cos(yaw) * cy
        ox, oy, oth = ODOM_OFFSET
        c, s = math.cos(oth), math.sin(oth)
        xm, ym = c * (xo - ox) + s * (yo - oy), -s * (xo - ox) + c * (yo - oy)
        dist = np.array([obs.clearance((a, b), n.world_all) for a, b in zip(xm, ym)]) if len(xm) else np.zeros(0)
        on = float((dist < 0.45).mean()) if len(dist) else float('nan')
        print(f'map: {n.maps} msgs, frame {g.header.frame_id}, {g.info.width}x{g.info.height} cells @ '
              f'{g.info.resolution:.2f} m ({len(g.data) / 1000:.1f} kB), seen {int((d >= 0).sum())}, occupied {len(rows)}: '
              f'{100 * on:.0f}% within 0.45 m of a real wall or plant', flush=True)
    mk = n.markers_last
    if mk is None:
        print('objects: NOTHING on /erc/explore/objects', flush=True)
    else:
        labels = [(m.text, round(m.pose.position.x, 1), round(m.pose.position.y, 1)) for m in mk.markers if m.text]
        # back to the mission frame, to compare with where the plants are
        ox, oy, oth = ODOM_OFFSET
        c, s = math.cos(oth), math.sin(oth)
        back = [(t, round(c * (x - ox) + s * (y - oy), 1), round(-s * (x - ox) + c * (y - oy), 1)) for t, x, y in labels]
        print(f'objects: {len(labels)} markers, in the mission frame: {back}', flush=True)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
