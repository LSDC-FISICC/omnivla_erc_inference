"""Fake rover + fake SDK (used by run_e2e.sh; see ../README.md) for an end-to-end test of checkpoint_controller_node (local_replan)
+ carrot_controller_node (obstacle_sidestep). The rover integrates /cmd_vel (the
checkpoint node's shaped output) with the measured plant; /erc/free_space comes from
test/obstacles.py. The SDK serves one checkpoint and accepts any arrival.

HEADING_SRC=gyro_gps: no /erc/heading_deg from here -- the raw sensors the SDK gives
instead, for Earth-rover-ros2-bridge's heading_node.py (source gyro_gps) to estimate it:
/erc/imu in bursts of 5 samples 20 ms apart every 0.5 s (bias GYRO_BIAS_DPS, default
Botswana's 0.76 deg/s; each burst's rate off by GYRO_BURST_ERR relative, default 0.25,
which gives the +-9% per turn measured on magcal_0917), /erc/wheel_twist, /erc/gps
updating at 1 Hz with GPS_SIGMA_M noise, GPS_LAG_S (1.5) old at its stamp and float32
lat/lon, and /erc/mags: MAG_MODE=frozen (default, Botswana's: it never turns) or good (the
true heading with a +-6 deg first harmonic and 2 deg noise, like a valid calibration; zero
declination, no hard iron). START_YAW_DEG: initial heading (ENU, 0 = east, the route's
direction in most worlds). HEADING_SRC=fused runs heading_node's source fused; run_e2e.sh
gives it a valid calibration file with MAG_CAL=valid."""
import collections
import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rclpy
from rclpy.time import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import Log
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Imu, LaserScan, MagneticField, NavSatFix
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
    # mission_24sept_Nav2_circles: the leg starts with the route ~70 deg off the heading, a planter
    # alongside; that is where Nav2 circled with the follower's gain at 0.36 on a 1.23 unit
    'corner': ([obs.Segment((-1.5, 2.0), (-1.5, 9.0)), obs.Segment((2.5, 1.0), (4.5, 1.0))], (4.0, 14.0)),
    # park beds OSM does not have (controller_sim park scenarios, 28-sept): a 10 x 8 m bed on the
    # route, and a 2 m sidewalk with a bench that leaves a 1.2 m gap
    'garden_10x8': ([obs.Segment((7, -4), (17, -4)), obs.Segment((17, -4), (17, 4)),
                     obs.Segment((17, 4), (7, 4)), obs.Segment((7, 4), (7, -4))], (30.0, 0.0)),
    'sidewalk_2_bench': ([obs.Segment((-2, 1.0), (28, 1.0)), obs.Segment((-2, -1.0), (28, -1.0)),
                          obs.Segment((10, 0.2), (11.5, 0.2)), obs.Segment((11.5, 0.2), (11.5, 1.0)),
                          obs.Segment((10, 0.2), (10, 1.0))], (28.0, 0.0)),
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
    """One checkpoint at the goal; START_CHECKPOINT=1 puts another first, where the rover starts
    (mission_botswana's first checkpoint was confirmed 6 s in, without moving)."""
    points = ([(0.0, 0.0)] if os.environ.get('START_CHECKPOINT') == '1' else []) + [goal]
    cps = [{'id': i + 1, 'sequence': i + 1, 'latitude': latlon(*p)[0], 'longitude': latlon(*p)[1]}
           for i, p in enumerate(points)]
    reached = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path.startswith('/checkpoints-list'):
                body = {'checkpoints_list': cps, 'latest_scanned_checkpoint': len(reached)}
            else:
                reached.append(len(reached) + 1)
                STATE.setdefault('reached_t', []).append(STATE.get('t', 0.0))
                if len(reached) >= len(cps):
                    STATE['done'] = True
                    body = {'mission_completed': True, 'message': 'fake SDK: done'}
                else:
                    body = {'mission_completed': False, 'next_checkpoint_sequence': len(reached) + 1}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(data)
    # threaded: 8765 is also foxglove_bridge's port, and a Foxglove client retrying through VS Code's
    # port forwarding held a single-threaded server, so a checkpoint POST timed out (29-sept, 2 of 66)
    srv = ThreadingHTTPServer(('127.0.0.1', 8765), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


PLANT_KW_MOVING = float(os.environ.get('PLANT_KW_MOVING', 0.36))
PLANT_KW_IN_PLACE = float(os.environ.get('PLANT_KW_IN_PLACE', 1.18))
HEADING_SRC = os.environ.get('HEADING_SRC', 'truth')
START_YAW = math.radians(float(os.environ.get('START_YAW_DEG', 0.0)))
GYRO_BIAS = math.radians(float(os.environ.get('GYRO_BIAS_DPS', 0.76)))
GYRO_BURST_ERR = float(os.environ.get('GYRO_BURST_ERR', 0.25))
GPS_SIGMA_M = float(os.environ.get('GPS_SIGMA_M', 0.4))
GPS_LAG_S = float(os.environ.get('GPS_LAG_S', 1.5))
MAG_MODE = os.environ.get('MAG_MODE', 'frozen')
RAW_SENSORS = HEADING_SRC in ('gyro_gps', 'fused')


class Fake(Node):
    def __init__(self, world):
        super().__init__('fake_world')
        self.world, self.goal = WORLDS[world]
        self.x = self.y = 0.0
        self.th = START_YAW
        self.v = self.w = 0.0
        self.buf = collections.deque([(0.0, 0.0)] * 26)   # 1.3 s at 20 Hz
        self.cmd = (0.0, 0.0)
        self.rng = np.random.default_rng(1)
        self.min_clear, self.hit, self.t = 9.0, False, 0.0
        self.events, self.replans = [], 0
        self.gps = self.create_publisher(NavSatFix, '/erc/gps/filtered', 10)
        self.hdg = self.create_publisher(Float32, '/erc/heading_deg', 10)
        self.scan = self.create_publisher(LaserScan, '/erc/free_space', 10)
        # body-frame velocity, as ekf_local publishes it: MPPI starts from it
        self.odom = self.create_publisher(Odometry, '/erc/odometry/local', 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.create_subscription(String, '/omnivla_debug', self._dbg, 10)
        self.create_subscription(Log, '/rosout', self._log, 50)
        self.create_subscription(Path, '/erc/local_route', self._route, LATCHED)
        self.create_timer(0.05, self._step)
        self.create_timer(1 / 3, self._pub_scan)
        self.create_timer(0.1, self._pub_pose)
        if RAW_SENSORS:
            self.hist = collections.deque(maxlen=200)      # (stamp s, w, v, x, y), 10 s
            self.imu = self.create_publisher(Imu, '/erc/imu', 50)
            self.wheels = self.create_publisher(TwistWithCovarianceStamped, '/erc/wheel_twist', 50)
            self.raw_gps = self.create_publisher(NavSatFix, '/erc/gps', 10)
            self.mags = self.create_publisher(MagneticField, '/erc/mags', 10)
            self.gps_fix, self.gps_next = None, 0.0
            self.create_timer(0.5, self._pub_sdk)
            self.hdg_err = []                       # (t, estimated - true heading, deg)
            self.create_subscription(Float32, '/erc/heading_deg', self._hdg_est, 10)

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

    def _hdg_est(self, m):
        true = (90.0 - math.degrees(self.th)) % 360.0
        self.hdg_err.append((self.t, (m.data - true + 180.0) % 360.0 - 180.0))

    def _log(self, m):
        if (m.name == 'heading_node' and 'gyro_gps:' in m.msg) or (
                m.name == 'checkpoint_controller_node' and ('Heading' in m.msg or 'anchor' in m.msg)):
            self.events.append((round(self.t, 1), ('HN ' if m.name == 'heading_node' else 'CP ') + m.msg[:150]))
            return
        if m.name == 'checkpoint_controller_node' and ('[local-plan]' in m.msg or 'Local replanning' in m.msg
                                                       or 'Navigating' in m.msg or 'Mission' in m.msg):
            if '[local-plan]' in m.msg and 'rejoin' in m.msg:
                self.replans += 1
            self.events.append((round(self.t, 1), 'CP ' + m.msg[:150]))

    def _step(self):
        self.t += 0.05
        STATE['t'] = self.t
        self.buf.append(self.cmd)
        v, w = self.buf.popleft()
        # measured on mission_16sept (0.36 moving); the Wuhan units measure 1.05-1.26 moving
        # (1.23 in mission_24sept_Nav2_circles): PLANT_KW_MOVING / PLANT_KW_IN_PLACE
        kw = PLANT_KW_IN_PLACE if abs(v) < 0.05 else PLANT_KW_MOVING
        self.v += (1.11 * v - self.v) * min(1, 0.05 / 0.47)
        self.w += (kw * w - self.w) * min(1, 0.05 / 0.35)
        self.th += self.w * 0.05
        self.turned = getattr(self, 'turned', 0.0) + abs(self.w) * 0.05
        self.x += self.v * math.cos(self.th) * 0.05
        self.y += self.v * math.sin(self.th) * 0.05
        c = obs.clearance((self.x, self.y), self.world)
        self.min_clear = min(self.min_clear, c)
        self.hit |= c <= obs.FOOTPRINT_W / 2
        if RAW_SENSORS:
            self.hist.append((self.get_clock().now().nanoseconds * 1e-9, self.w, self.v, self.x, self.y))

    def _at(self, t, k):
        ts = [h[0] for h in self.hist]
        return float(np.interp(t, ts, [h[k] for h in self.hist]))

    def _pub_sdk(self):
        """One SDK poll: the last 5 gyro/wheel samples, the GPS fix, the magnetometer."""
        if len(self.hist) < 5:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        gain = 1.0 + GYRO_BURST_ERR * self.rng.standard_normal()
        for k in range(5):
            t = now - 0.02 * (4 - k)
            stamp = Time(nanoseconds=int(t * 1e9)).to_msg()
            m = Imu()
            m.header.stamp, m.header.frame_id = stamp, 'imu_link'
            m.angular_velocity.z = (gain * self._at(t, 1) + GYRO_BIAS
                                    + math.radians(0.1) * self.rng.standard_normal())
            m.linear_acceleration.z = 9.81
            self.imu.publish(m)
            tw = TwistWithCovarianceStamped()
            tw.header.stamp, tw.header.frame_id = stamp, 'base_link'
            v = self._at(t, 2)
            tw.twist.twist.linear.x = v + (0.01 * self.rng.standard_normal() if abs(v) > 0.01 else 0.0)
            self.wheels.publish(tw)
        if now >= self.gps_next:
            # the GPS updates at 1 Hz; the SDK repeats the fix between updates
            self.gps_next = now + 1.0
            t = now - GPS_LAG_S
            x = self._at(t, 3) + GPS_SIGMA_M * self.rng.standard_normal()
            y = self._at(t, 4) + GPS_SIGMA_M * self.rng.standard_normal()
            self.gps_fix = tuple(float(np.float32(v)) for v in latlon(x, y))
        f = NavSatFix()
        f.header.stamp, f.header.frame_id = self.get_clock().now().to_msg(), 'gps_link'
        f.status.status = 0
        f.latitude, f.longitude = self.gps_fix
        self.raw_gps.publish(f)
        mg = MagneticField()
        mg.header.stamp = self.get_clock().now().to_msg()
        if MAG_MODE == 'good':
            # heading_node: mapped = (-y_raw, x_raw, z), psi = atan2(mapped_x, mapped_y)
            psi = self.th + math.radians(6.0) * math.sin(self.th + 0.7) + math.radians(2.0) * self.rng.standard_normal()
            mg.magnetic_field.x, mg.magnetic_field.y, mg.magnetic_field.z = 30e-6 * math.cos(psi), -30e-6 * math.sin(psi), -20e-6
        else:                         # frozen, like Botswana's: it never sees the rover turn
            mg.magnetic_field.x, mg.magnetic_field.y, mg.magnetic_field.z = 150e-6, -60e-6, 40e-6
        self.mags.publish(mg)

    def _pub_pose(self):
        f = NavSatFix()
        f.latitude, f.longitude = latlon(self.x, self.y)
        self.gps.publish(f)
        if not RAW_SENSORS:
            self.hdg.publish(Float32(data=float((90.0 - math.degrees(self.th)) % 360.0)))
        elif self.t > 0.1:
            self.true_hdg = (90.0 - math.degrees(self.th)) % 360.0
        o = Odometry()
        o.header.stamp = self.get_clock().now().to_msg()
        o.header.frame_id, o.child_frame_id = 'odom', 'base_link'
        o.twist.twist.linear.x, o.twist.twist.angular.z = float(self.v), float(self.w)
        self.odom.publish(o)

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
    if RAW_SENSORS and n.hdg_err:
        e = np.array(n.hdg_err)
        anc = [ev[0] for ev in n.events if 'heading anchored' in ev[1] or 'heading confirmed' in ev[1]]
        after = np.abs(e[e[:, 0] > anc[0], 1]) if anc else np.zeros(0)
        print(f'heading ({HEADING_SRC}, mag {MAG_MODE}) vs truth: anchored at t={anc[0] if anc else float("nan"):.1f}s; '
              f'before |err| p50 {np.median(np.abs(e[e[:, 0] <= (anc[0] if anc else 1e9), 1])):.0f} deg; after: '
              + (f'|err| p50 {np.median(after):.1f} p90 {np.percentile(after, 90):.1f} max {after.max():.1f} deg'
                 if len(after) else 'n/a'))
    d = math.hypot(n.x - n.goal[0], n.y - n.goal[1])
    print(f'{world}: t={n.t:.0f}s final=({n.x:.1f},{n.y:.1f}) goal_dist={d:.1f}m min_clear={n.min_clear:.2f}m '
          f'hit={n.hit} completed={STATE["done"]} checkpoints_at={[round(x) for x in STATE.get("reached_t", [])]}s '
          f'local_replans={n.replans} '
          f'turned={math.degrees(getattr(n, "turned", 0.0)):.0f}deg ({math.degrees(getattr(n, "turned", 0.0)) / 360:.1f} full turns) '
          f'plant_kw_moving={PLANT_KW_MOVING}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
