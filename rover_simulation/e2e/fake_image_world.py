"""Fake NYU indoor world for an end-to-end test of image_checkpoint_controller_node (run_e2e_images.sh).

The REAL mission node and carrot_controller_node run against:
  - the plant of fake_world.py (1.3 s command delay, k_v 1.11, k_w PLANT_KW_MOVING / 1.18 in
    place), integrated on the wall clock, with dead-reckoning errors on the published odometry
    (WHEEL_SCALE, GYRO_SCALE, GYRO_BIAS_DEG_MIN) and an odom origin offset from the track;
  - the NYU walls and tables (test/indoor_sim.py), chairs and open doors (test/loop_patrol_sim.py),
    cones placed at random along the loop (seed), the orange start cone behind the rover;
  - /erc/free_space from test/obstacles.profile, and /erc/cones as cone_detector_node publishes
    it (<= 10 m, +-55 deg, p 0.9, occluded, 0.7 deg / 3 % noise), both from where the rover
    was IMAGE_LAG_S ago;
  - a fake SDK /checkpoint-reached: 200 only if the rover really is within 2 m of the cone of
    the next goal (CP1, CP2, CP3, CP4, Finish), else 400.

    python3 fake_image_world.py <seconds> <seed> <layout loop|random> <world tables|chairs|chairs+doors>
"""
import collections
import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Log
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'erc_inference', 'test'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'erc_inference'))
import cone_sim  # noqa: E402
import image_goal_sim as igs  # noqa: E402
import indoor_sim as isim  # noqa: E402
import loop_patrol_sim as lps  # noqa: E402
import obstacles as obs  # noqa: E402

PLANT_KW_MOVING = float(os.environ.get('PLANT_KW_MOVING', 1.1))
PLANT_KW_IN_PLACE = float(os.environ.get('PLANT_KW_IN_PLACE', 1.18))
WHEEL_SCALE = float(os.environ.get('WHEEL_SCALE', 0.01))
GYRO_SCALE = float(os.environ.get('GYRO_SCALE', 0.01))
GYRO_BIAS_DEG_MIN = float(os.environ.get('GYRO_BIAS_DEG_MIN', 0.5))
IMAGE_LAG_S = float(os.environ.get('IMAGE_LAG_S', 0.8))
SDK_PORT = int(os.environ.get('SDK_PORT', 8766))
ODOM_OFFSET = (5.0, -3.0, math.radians(30.0))
ORDER = ['CP1', 'CP2', 'CP3', 'CP4', 'Finish']
CLS = {'CP1': 'red_orange', 'CP2': 'blue', 'CP3': 'green', 'CP4': 'yellow', 'Finish': 'red_orange'}
STATE = {'next': 0, 'log': [], 'done': False}


class World(Node):
    def __init__(self, seed, layout, world):
        super().__init__('fake_image_world')
        rng = np.random.default_rng(seed)
        self.rng = rng
        if world in ('doors', 'chairs+doors'):
            walls, self.gaps = lps.walls_with_doors(rng)
        else:
            walls, self.gaps = isim.nyu_walls(), []
        if 'chairs' in world:
            walls = walls + lps.chairs(rng)
        cps = igs.place_cones(layout, rng, walls)
        self.cones = {'CP1': cps[0], 'CP2': cps[1], 'CP3': cps[2], 'CP4': cps[3], 'Finish': np.array([-0.8, 0.0])}
        self.walls = walls
        self.cone_objs = {n: obs.Disc(p, cone_sim.CONE_R) for n, p in self.cones.items()}
        self.world_all = walls + list(self.cone_objs.values())
        self.x = self.y = self.th = 0.0
        self.ex = self.ey = self.eth = 0.0
        self.v = self.w = 0.0
        self.cmds = collections.deque([(-1e9, 0.0, 0.0)])
        self.cmd = (0.0, 0.0)
        self.hist = collections.deque()
        self.t0 = self._now()
        self.last = None
        self.hits, self.hit_now, self.min_clear = [], False, 9.0
        self.pose_err = []
        self.est_pose = None
        self.odom = self.create_publisher(Odometry, '/erc/odometry/local', 10)
        self.scan = self.create_publisher(LaserScan, '/erc/free_space', 10)
        self.cone_pub = self.create_publisher(String, '/erc/cones', 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.create_subscription(PoseStamped, '/erc/indoor/pose', self._pose, 10)
        self.create_subscription(Log, '/rosout', self._log, 50)
        self.create_timer(0.05, self._step)
        self.create_timer(1 / 30, self._pub_odom)
        self.create_timer(1 / 3, self._camera)
        STATE['world'] = self
        print('cones: ' + ', '.join(f'{k} ({v[0]:.1f},{v[1]:.1f})' for k, v in self.cones.items())
              + f' | doors {self.gaps}', flush=True)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def t(self):
        return self._now() - self.t0

    def _cmd(self, m):
        self.cmds.append((self._now(), m.linear.x, m.angular.z))

    def _pose(self, m):
        self.est_pose = (m.pose.position.x, m.pose.position.y)

    def _log(self, m):
        if m.name == 'image_checkpoint_controller_node' and (
                'confirmed' in m.msg or 'Mission' in m.msg or 'cone on the map' in m.msg or 'goal ' in m.msg[:6]
                or 'other way' in m.msg or 'arrived' in m.msg):
            print(f'  {self.t():6.1f}s  {m.msg[:150]}', flush=True)

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
        self.x += self.v * math.cos(self.th) * dt
        self.y += self.v * math.sin(self.th) * dt
        self.eth += (self.w * (1 + GYRO_SCALE) + math.radians(GYRO_BIAS_DEG_MIN) / 60.0) * dt
        self.ex += self.v * (1 + WHEEL_SCALE) * math.cos(self.eth) * dt
        self.ey += self.v * (1 + WHEEL_SCALE) * math.sin(self.eth) * dt
        self.hist.append((now, self.x, self.y, self.th))
        while len(self.hist) > 1 and self.hist[1][0] <= now - IMAGE_LAG_S:
            self.hist.popleft()
        c = obs.clearance((self.x, self.y), self.world_all)
        self.min_clear = min(self.min_clear, c)
        h = c <= obs.FOOTPRINT_W / 2
        if h and not self.hit_now:
            self.hits.append((round(self.t(), 1), round(self.x, 1), round(self.y, 1)))
        self.hit_now = h
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
        for n, cpos in self.cones.items():
            d = cpos - o
            r = float(np.hypot(*d))
            brg = math.degrees(math.remainder(math.atan2(d[1], d[0]) - lth, 2 * math.pi))
            if abs(brg) > 55.0 or r > 10.0 or self.rng.random() > 0.9:
                continue
            if not cone_sim.visible(o, cpos, self.walls + [c for k, c in self.cone_objs.items() if k != n], 10.0):
                continue
            dets.append({'color': CLS[n], 'bearing_deg': brg + self.rng.normal(0, 0.7),
                         'range_m': r * (1 + self.rng.normal(0, 0.03)), 'consistent': True})
        self.cone_pub.publish(String(data=json.dumps({'stamp': self._now(), 'detections': dets})))


def sdk():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            w = STATE.get('world')
            body, code = {}, 200
            if self.path.startswith('/checkpoints-list'):
                body = {'checkpoints_list': [{'id': i + 1, 'sequence': i + 1} for i in range(5)],
                        'latest_scanned_checkpoint': STATE['next']}
            elif w is not None and STATE['next'] < len(ORDER):
                name = ORDER[STATE['next']]
                d = math.hypot(w.x - w.cones[name][0], w.y - w.cones[name][1])
                if d <= 2.0:
                    STATE['next'] += 1
                    STATE['log'].append((round(w.t(), 1), name, round(d, 2)))
                    done = STATE['next'] == len(ORDER)
                    STATE['done'] = done
                    body = {'mission_completed': done, 'message': f'{name} reached at {d:.2f} m',
                            'next_checkpoint_sequence': STATE['next'] + 1}
                    print(f'  {w.t():6.1f}s  SDK: {name} accepted, {d:.2f} m from the cone', flush=True)
                else:
                    code = 400
                    body = {'message': f'{name} is {d:.1f} m away'}
                    print(f'  {w.t():6.1f}s  SDK: rejected, {name} is {d:.1f} m away', flush=True)
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(data)
    srv = HTTPServer(('127.0.0.1', SDK_PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main():
    dur, seed, layout, world = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    sdk()
    rclpy.init()
    n = World(seed, layout, world)
    while rclpy.ok() and n.t() < dur and not STATE['done']:
        rclpy.spin_once(n, timeout_sec=0.01)
    pe = np.array(n.pose_err) if n.pose_err else np.array([np.nan])
    print(f'images_e2e seed={seed} layout={layout} world={world}: t={n.t():.0f}s accepted={len(STATE["log"])}/5 '
          f'{STATE["log"]} hits={len(n.hits)} {n.hits[:6]} min_clear={n.min_clear:.2f} '
          f'pose_err p50={np.nanmedian(pe):.2f} max={np.nanmax(pe):.2f}', flush=True)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
