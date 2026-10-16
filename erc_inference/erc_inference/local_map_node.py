#!/usr/bin/env python3
"""The 2D map the rover would build, with nobody driving: something to LOOK at.

The exploring mission (explore_search, in image_checkpoint_controller_node) builds its map
from /erc/free_space and puts the objects of /erc/objects on it -- but only once the mission
has been started, and then the rover moves. This node builds the same map, with the same
class and the same indoor parameters (local_planner.LocalCostmap, loop_patrol.LOCAL_INDOOR,
object_tracker.ObjectTracker), from the same two topics and /erc/odometry/local, and never commands
anything: push the rover by hand or drive it from the SDK panel and watch what it would see.

    /erc/local_map          nav_msgs/OccupancyGrid, odometry frame, the seen part only
    /erc/local_map/objects  visualization_msgs/MarkerArray: the objects (potted plants)
    /erc/local_map/path     nav_msgs/Path: where the rover has been

It is not the mission's map: that one starts when the mission does and is on
/erc/explore/map. While the mission publishes its objects this node keeps its own markers
back, so they are not drawn twice.

    ros2 run erc_inference local_map_node
    ros2 service call /local_map_node/reset std_srvs/srv/Empty     # start the map again from here
"""
import collections
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty
from visualization_msgs.msg import MarkerArray

from erc_inference import local_planner as lp
from erc_inference.loop_patrol import DEFAULTS as PATROL_DEFAULTS
from erc_inference.loop_patrol import LOCAL_INDOOR
from erc_inference.people_mask import PeopleMask
from erc_inference.object_tracker import ObjectTracker
from erc_inference.map_viz import grid_message, object_markers


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LocalMapNode(Node):
    def __init__(self):
        super().__init__('local_map_node')
        self.declare_parameter('odom_topic', '/erc/odometry/local')
        self.declare_parameter('free_space_topic', '/erc/free_space')
        self.declare_parameter('objects_topic', '/erc/objects')
        self.declare_parameter('object_classes', ['potted plant'])
        self.declare_parameter('person_classes', ['person'])     # kept off the map, as the mission does
        self.declare_parameter('map_topic', '/erc/local_map')
        self.declare_parameter('mission_objects_topic', '/erc/explore/objects')
        self.declare_parameter('map_half_m', 30.0)           # the grid: a square this far each way, kept around the rover
        self.declare_parameter('map_forget_m', 10.0)         # a cell is dropped this many metres DRIVEN after it was
                                                             # last seen (0: never): the pose drifts, old map lies
        self.declare_parameter('map_radius_m', 5.0)          # ... and when further than this from the rover (0: kept)
        self.declare_parameter('publish_rate_hz', 1.0)
        for name, default in LOCAL_INDOOR.items():
            self.declare_parameter(f'local.{name}', default)
        self._hist = collections.deque()                     # (stamp s, x, y, yaw), last 5 s
        self._replanner = None
        self._objects = None
        self._trail = []
        self._mission_markers_t = -1e9
        self._frame = None
        self._held_back = False
        g = lambda n: self.get_parameter(n).value
        self._map_pub = self.create_publisher(OccupancyGrid, g('map_topic'), 1)
        self._obj_pub = self.create_publisher(MarkerArray, g('map_topic') + '/objects', 1)
        self._path_pub = self.create_publisher(Path, g('map_topic') + '/path', 1)
        self.create_subscription(Odometry, g('odom_topic'), self._on_odom, 50)
        self.create_subscription(LaserScan, g('free_space_topic'), self._on_scan, 10)
        self.create_subscription(String, g('objects_topic'), self._on_objects, 10)
        self.create_subscription(MarkerArray, g('mission_objects_topic'), self._on_mission_markers, 1)
        self.create_service(Empty, '~/reset', self._on_reset)
        self.create_timer(1.0 / float(g('publish_rate_hz')), self._publish)
        self.get_logger().info(f"local map (view only, commands nothing): {g('free_space_topic')} + "
                               f"{g('objects_topic')} on {g('odom_topic')} -> {g('map_topic')}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _start(self, x, y):
        h = float(self.get_parameter('map_half_m').value)
        kw = {n: self.get_parameter(f'local.{n}').value for n in LOCAL_INDOOR}
        kw['margin_m'] = 0.0
        self._replanner = lp.LocalReplanner(np.array([[x - h, y - h], [x + h, y + h]]), **kw)
        self._objects = ObjectTracker()
        self._people = PeopleMask()
        self._trail = []
        self._odometer, self._odo_xy, self._forgot_at = 0.0, (x, y), 0.0
        self._frame = None

    def _on_reset(self, _req, res):
        self._replanner = None            # started again at the next odometry sample
        self.get_logger().info('map reset')
        return res

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        x, y, yaw = p.position.x, p.position.y, _yaw(p.orientation)
        if self._replanner is None:
            self._start(x, y)
        self._frame = msg.header.frame_id or 'odom'
        self._hist.append((t, x, y, yaw))
        while self._hist and t - self._hist[0][0] > 5.0:
            self._hist.popleft()
        if not self._trail or math.hypot(x - self._trail[-1][0], y - self._trail[-1][1]) > 0.1:
            self._trail.append((x, y, yaw, self._odometer))
        self._travel(x, y)

    def _travel(self, x, y):
        """Metres driven; every metre: the grid stays around the rover and what was last seen
        map_forget_m of driving ago is dropped (map, objects and path alike)."""
        d = math.hypot(x - self._odo_xy[0], y - self._odo_xy[1])
        if d < 1.0:
            self._odometer += d
        self._odo_xy = (x, y)
        if self._odometer - self._forgot_at < 0.5:
            return
        self._forgot_at = self._odometer
        m = self._replanner.map
        radius = float(self.get_parameter('map_radius_m').value)
        if radius > 0.0:
            m.forget_beyond(x, y, radius)
        half = 0.5 * m.w * m.p['resolution_m']
        if max(abs(x - (m.ox + half)), abs(y - (m.oy + half))) > 0.5 * half:
            m.recenter(x, y)
        forget = float(self.get_parameter('map_forget_m').value)
        if forget > 0.0:
            m.forget(self._odometer - forget)
            self._objects.forget(self._odometer - forget)
            self._trail = [p for p in self._trail if p[3] >= self._odometer - forget]

    def _pose_at(self, stamp):
        """The pose at a perception stamp (nearest odometry sample within 0.5 s), else the latest."""
        if not self._hist:
            return None
        if stamp <= 0.0:
            return self._hist[-1]
        best = min(self._hist, key=lambda h: abs(h[0] - stamp))
        return best if abs(best[0] - stamp) < 0.5 else self._hist[-1]

    def _turn_rate_dps(self):
        h = [e for e in self._hist if self._hist[-1][0] - e[0] <= 1.0]
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.3:
            return 0.0
        d = (h[-1][3] - h[0][3] + math.pi) % (2 * math.pi) - math.pi
        return abs(math.degrees(d)) / (h[-1][0] - h[0][0])

    def _on_scan(self, msg: LaserScan):
        pose = self._pose_at(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)
        if pose is None or self._replanner is None:
            return
        r = np.asarray(msg.ranges, float)
        b = np.degrees(msg.angle_min + np.arange(len(r)) * msg.angle_increment)
        hit = np.isfinite(r) & (r < msg.range_max - 0.05)
        r = np.where(np.isfinite(r), r, msg.range_max)
        r, hit = self._people.filter(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, pose[3], b, r, hit)
        # the planner's own gate: scans are skipped while the heading turns faster than it can trust
        self._replanner.map.odometer = self._odometer
        self._replanner.observe(pose[0], pose[1], pose[2], pose[3], b, r, hit)

    def _on_objects(self, msg: String):
        try:
            doc = json.loads(msg.data)
        except ValueError:
            return
        pose = self._pose_at(float(doc.get('stamp', 0.0)))
        if pose is None or self._objects is None:
            return
        persons = set(self.get_parameter('person_classes').value)
        self._people.observe(float(doc.get('stamp', 0.0)), pose[1], pose[2], pose[3], [
            (float(d['bearing_right_deg']), float(d['bearing_left_deg']),
             d['range_m'] if d.get('range_m') is not None else d.get('range_scan_m'))
            for d in doc.get('detections', []) if d.get('label') in persons and 'bearing_left_deg' in d],
            self._replanner.map if self._replanner is not None else None)
        if self._turn_rate_dps() > PATROL_DEFAULTS['cone_max_turn_dps']:
            return                        # the image is older than the heading: a turning rover misplaces it
        wanted = set(self.get_parameter('object_classes').value)
        frame = [(d['label'], float(d['bearing_deg']), float(d['range_m'])) for d in doc.get('detections', [])
                 if d.get('label') in wanted and d.get('range_m') is not None
                 and float(d['range_m']) <= PATROL_DEFAULTS['max_detection_m']]
        self._objects.observe(pose[1], pose[2], pose[3], frame, self._now(), self._odometer)   # the frame at once

    def _on_mission_markers(self, _msg):
        self._mission_markers_t = self._now()

    def _publish(self):
        if self._replanner is None or self._frame is None:
            return
        stamp = self.get_clock().now().to_msg()
        grid = grid_message(self._replanner.map, stamp, self._frame)
        if grid is not None:
            self._map_pub.publish(grid)
        mission = self._now() - self._mission_markers_t < 3.0
        objs = [] if mission else [
            (f'{c.cls} #{c.id}', float(c.estimate()[0]), float(c.estimate()[1]), False, len(c.points))
            for c in self._objects.clusters if c.confirmed]
        if not mission or not self._held_back:
            self._obj_pub.publish(object_markers(objs, stamp, self._frame))     # empty once, then silent
        self._held_back = mission
        path = Path()
        path.header.stamp, path.header.frame_id = stamp, self._frame
        for x, y, yaw, _odo in self._trail[-2000:]:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            path.poses.append(ps)
        self._path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = LocalMapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
