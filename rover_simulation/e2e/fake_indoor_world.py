"""Fake rover in the NYU indoor loop (used by run_e2e_indoor.sh) for an end-to-end test
of indoor_mission_node + a model node stand-in (carrot_controller_node) in ROS.

Same plant as fake_world.py: integrates /cmd_vel (indoor_mission_node's shaped
output) with 1.3 s of delay and the measured gains. Unlike fake_world.py it
publishes the Odometry POSE, as ekf_local does -- indoor_mission_node anchors
it -- with optional dead-reckoning errors (env WHEEL_SCALE, GYRO_BIAS_DEG_MIN).
Walls: test/indoor_sim.py nyu_walls(). No SDK: run the mission node with
confirm_with_sdk:=false.

    python3 fake_indoor_world.py <seconds>
"""
import collections
import math
import os
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Log
from rclpy.node import Node

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'erc_inference', 'test'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'erc_inference'))
import obstacles as obs  # noqa: E402
from indoor_sim import nyu_walls  # noqa: E402

PLANT_KW_MOVING = float(os.environ.get('PLANT_KW_MOVING', 0.36))
PLANT_KW_IN_PLACE = float(os.environ.get('PLANT_KW_IN_PLACE', 1.18))
WHEEL_SCALE = float(os.environ.get('WHEEL_SCALE', 0.0))
GYRO_BIAS_DEG_MIN = float(os.environ.get('GYRO_BIAS_DEG_MIN', 0.0))
# ekf_local's odom frame starts wherever the EKF started, not at the track
# origin: an offset here checks that indoor_mission_node anchors instead of
# assuming odom == track.
ODOM_OFFSET = (5.0, -3.0, math.radians(30.0))
STATE = {'done': False, 'confirmed': []}


class FakeIndoor(Node):
    def __init__(self):
        super().__init__('fake_indoor_world')
        self.world = nyu_walls()
        self.x = self.y = self.th = 0.0            # truth, track frame
        self.ex = self.ey = self.eth = 0.0         # dead-reckoned, track frame
        self.v = self.w = 0.0
        self.buf = collections.deque([(0.0, 0.0)] * 26)   # 1.3 s at 20 Hz
        self.cmd = (0.0, 0.0)
        self.min_clear, self.hit, self.t = 9.0, False, 0.0
        self.hits = []
        self.odom = self.create_publisher(Odometry, '/erc/odometry/local', 10)
        self.create_subscription(Twist, '/cmd_vel', lambda m: setattr(self, 'cmd', (m.linear.x, m.angular.z)), 10)
        self.create_subscription(Log, '/rosout', self._log, 50)
        self.create_timer(0.05, self._step)
        self.create_timer(1 / 30, self._pub_odom)

    def _log(self, m):
        if m.name == 'indoor_mission_node':
            if 'confirmed' in m.msg:
                STATE['confirmed'].append((round(self.t, 1), m.msg.split(':')[0], round(self.x, 1), round(self.y, 1)))
            if 'Mission failed' in m.msg or 'Leg to' in m.msg:
                print(f'  {self.t:6.1f}s  {m.msg[:140]}', flush=True)

    def _step(self):
        self.t += 0.05
        self.buf.append(self.cmd)
        v, w = self.buf.popleft()
        kw = PLANT_KW_IN_PLACE if abs(v) < 0.05 else PLANT_KW_MOVING
        self.v += (1.11 * v - self.v) * min(1, 0.05 / 0.47)
        self.w += (kw * w - self.w) * min(1, 0.05 / 0.35)
        self.th += self.w * 0.05
        self.x += self.v * math.cos(self.th) * 0.05
        self.y += self.v * math.sin(self.th) * 0.05
        self.eth += (self.w + math.radians(GYRO_BIAS_DEG_MIN) / 60.0) * 0.05
        self.ex += self.v * (1 + WHEEL_SCALE) * math.cos(self.eth) * 0.05
        self.ey += self.v * (1 + WHEEL_SCALE) * math.sin(self.eth) * 0.05
        c = obs.clearance((self.x, self.y), self.world)
        self.min_clear = min(self.min_clear, c)
        if c <= obs.FOOTPRINT_W / 2 and not self.hit:
            self.hits.append((round(self.t, 1), round(self.x, 2), round(self.y, 2)))
        self.hit = c <= obs.FOOTPRINT_W / 2

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


def main():
    dur = float(sys.argv[1])
    rclpy.init()
    n = FakeIndoor()
    while rclpy.ok() and n.t < dur and not (STATE['confirmed'] and 'Finish' in STATE['confirmed'][-1][1]):
        rclpy.spin_once(n, timeout_sec=0.05)
    for e in STATE['confirmed']:
        print(f'  {e[0]:6.1f}s  {e[1]} at true ({e[2]}, {e[3]})')
    print(f'nyu_indoor: t={n.t:.0f}s final=({n.x:.1f},{n.y:.1f}) confirmed={len(STATE["confirmed"])}/5 '
          f'min_clear={n.min_clear:.2f}m wall_contacts={n.hits} plant_kw_moving={PLANT_KW_MOVING} '
          f'wheel_scale={WHEEL_SCALE} gyro_bias={GYRO_BIAS_DEG_MIN}deg/min')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
