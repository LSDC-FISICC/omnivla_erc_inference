"""Fake rover + fake SDK (used by run_e2e.sh; see ../README.md) for an end-to-end test of checkpoint_controller_node (local_replan)
+ carrot_controller_node (obstacle_sidestep). The rover integrates /cmd_vel (the
checkpoint node's shaped output) with the measured plant; /erc/free_space comes from
test/obstacles.py. The SDK serves one checkpoint and accepts any arrival."""
import collections
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import Log
from sensor_msgs.msg import LaserScan, NavSatFix
from std_msgs.msg import Float32, String

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'erc_inference', 'test'))
import obstacles as obs  # noqa: E402

LAT0, LON0 = 30.4825077, 114.3026199
LATCHED = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
WORLDS = {
    'hedge': ([obs.Segment((4.0, -4.0), (20.0, 1.2))], (26.0, 0.0)),
    'wall': ([obs.Segment((9.0, -6.0), (9.0, 2.0))], (20.0, 0.0)),
    'chicane': ([obs.Segment((8.0, -3.0), (8.0, 0.55)), obs.Segment((13.0, -0.55), (13.0, 3.0))], (24.0, 0.0)),
}
STATE = {'done': False}


K_NORTH = math.radians(1.0) * 6378137.0
K_EAST = K_NORTH * math.cos(math.radians(LAT0))


def latlon(x, y):
    """World metres (east, north) -> lat/lon. Equirectangular about LAT0/LON0, the
    frame the nodes use -- NOT UTM: at Wuhan UTM grid north is 1.37 deg off true
    north, which put every mapped obstacle ~0.36 m off at 15 m in a first version."""
    return LAT0 + y / K_NORTH, LON0 + x / K_EAST


def sdk(goal):
    lat, lon = latlon(*goal)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path.startswith('/checkpoints-list'):
                body = {'checkpoints_list': [{'id': 1, 'sequence': 1, 'latitude': lat, 'longitude': lon}],
                        'latest_scanned_checkpoint': 0}
            else:
                STATE['done'] = True
                body = {'mission_completed': True, 'message': 'fake SDK: done'}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(data)
    srv = HTTPServer(('127.0.0.1', 8765), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


class Fake(Node):
    def __init__(self, world):
        super().__init__('fake_world')
        self.world, self.goal = WORLDS[world]
        self.x = self.y = self.th = 0.0
        self.v = self.w = 0.0
        self.buf = collections.deque([(0.0, 0.0)] * 26)   # 1.3 s at 20 Hz
        self.cmd = (0.0, 0.0)
        self.rng = np.random.default_rng(1)
        self.min_clear, self.hit, self.t = 9.0, False, 0.0
        self.events, self.replans = [], 0
        self.gps = self.create_publisher(NavSatFix, '/erc/gps/filtered', 10)
        self.hdg = self.create_publisher(Float32, '/erc/heading_deg', 10)
        self.scan = self.create_publisher(LaserScan, '/erc/free_space', 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.create_subscription(String, '/omnivla_debug', self._dbg, 10)
        self.create_subscription(Log, '/rosout', self._log, 50)
        self.create_subscription(Path, '/erc/local_route', self._route, LATCHED)
        self.create_timer(0.05, self._step)
        self.create_timer(1 / 3, self._pub_scan)
        self.create_timer(0.1, self._pub_pose)

    def _cmd(self, m):
        self.cmd = (m.linear.x, m.angular.z)

    def _route(self, m):
        self.last_route = [(p.pose.position.x, p.pose.position.y) for p in m.poses]

    def _dbg(self, m):
        if '[side-step]' in m.data:
            note = m.data.split('[side-step]', 1)[1].split('|')[0].strip()
            k = note.split()[0]
            if k in ('brake', 'turn', 'resume', 'give', 'still') and 'ahead=' in note or k in ('resume', 'give', 'still') \
                    or note.startswith('turn left') or note.startswith('turn right'):
                self.events.append((round(self.t, 1), 'SS ' + note[:70]))

    def _log(self, m):
        if m.name == 'checkpoint_controller_node' and ('[local-plan]' in m.msg or 'Local replanning' in m.msg
                                                       or 'Navigating' in m.msg or 'Mission' in m.msg):
            if '[local-plan]' in m.msg and 'rejoin' in m.msg:
                self.replans += 1
            self.events.append((round(self.t, 1), 'CP ' + m.msg[:150]))

    def _step(self):
        self.t += 0.05
        self.buf.append(self.cmd)
        v, w = self.buf.popleft()
        kw = 1.18 if abs(v) < 0.05 else 0.36
        self.v += (1.11 * v - self.v) * min(1, 0.05 / 0.47)
        self.w += (kw * w - self.w) * min(1, 0.05 / 0.35)
        self.th += self.w * 0.05
        self.x += self.v * math.cos(self.th) * 0.05
        self.y += self.v * math.sin(self.th) * 0.05
        c = obs.clearance((self.x, self.y), self.world)
        self.min_clear = min(self.min_clear, c)
        self.hit |= c <= obs.FOOTPRINT_W / 2

    def _pub_pose(self):
        f = NavSatFix()
        f.latitude, f.longitude = latlon(self.x, self.y)
        self.gps.publish(f)
        self.hdg.publish(Float32(data=float((90.0 - math.degrees(self.th)) % 360.0)))

    def _pub_scan(self):
        b, free, caps = obs.profile(self.x, self.y, self.th, self.world, self.rng)
        # as free_space_node reports it: max range where nothing was found
        free = np.where(free < caps - 1e-3, free, 3.0)
        s = LaserScan()
        s.header.stamp = self.get_clock().now().to_msg()
        s.angle_min, s.angle_max = math.radians(b[0]), math.radians(b[-1])
        s.angle_increment = math.radians(b[1] - b[0])
        s.range_max = 3.0
        s.ranges = [float(x) for x in free]
        self.scan.publish(s)


def main():
    world, dur = sys.argv[1], float(sys.argv[2])
    sdk(WORLDS[world][1])
    rclpy.init()
    n = Fake(world)
    while rclpy.ok() and n.t < dur and not STATE['done']:
        rclpy.spin_once(n, timeout_sec=0.05)
    for e in n.events:
        print(f'  {e[0]:6.1f}s  {e[1]}')
    d = math.hypot(n.x - n.goal[0], n.y - n.goal[1])
    print(f'{world}: t={n.t:.0f}s final=({n.x:.1f},{n.y:.1f}) goal_dist={d:.1f}m min_clear={n.min_clear:.2f}m '
          f'hit={n.hit} completed={STATE["done"]} local_replans={n.replans}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
