#!/usr/bin/env python3
"""The mission controller without OmniVLA: polar control on the carrot, nothing else.

Why this exists. The replay of mission_wuhan_hard (2026-09-22) ran the recorded
ticks through MotionController under steering_source=plan and =carrot and got
the same oscillation to the decimal -- 5.4 sign changes of omega per minute
either way. The model's applied deviation had median 0.0 deg; the 8 deg
deadzone swallowed nearly all of it. It was not steering. NuevoPlanNavegacion
9 anticipated exactly this outcome, and 4.2 mode (a) is "log only".

Taking it out of the process also frees the GPU: DA3 for the free-space profile
costs 85 ms alone against 205 ms when it shares the device (PLAN_CAMPO 2.4).

What it does, identically to omnivla_edge_node minus the model:
  * goal in the robot frame from /erc/gps/filtered + /erc/heading_deg, via the
    same robot_frame_offset / goal_bearing_from_offset
  * motion_control.MotionController with steering_source forced to "carrot"
  * publishes /omnivla/cmd_vel, so checkpoint_controller_node gates it exactly
    as before, and /omnivla_debug in the SAME line format, so every analysis
    tool already written against that log keeps working on new bags

What it adds, OFF by default -- obstacle_stop:
  Brake, do not steer, when /erc/free_space (erc_perception) reports something
  close straight ahead. Not avoidance: the avoidance selectors tried in the
  simulator hit every obstacle in every scenario. Braking is what
  NuevoPlanNavegacion 4.3 prescribes ("detection brakes, it does not plan"),
  and it is the one response that does not need the rover to out-turn a 1.3 s
  loop. In controller_sim it stopped the rover 0.37-0.42 m short of the kerb,
  the post and the chicane in 15 of 15 runs -- and the mission then does not
  complete: a crash becomes a stop that needs an operator. That is the trade. The profile saw the 16-sept kerb at 0.20 m straight ahead before the
  rollover. It has never run in the field; turn it on deliberately.

What it adds, OFF by default -- obstacle_sidestep (sidestep.py):
  Instead of stopping for good: brake, turn ~60 deg IN PLACE towards the
  route's side if the profile shows it open, advance sidestep.advance_m, and hand back to the
  carrot, which brings the rover back to the route. Turning in place is the one
  thing this rover turns well at (k_w 1.18 against 0.36 moving). It replaces
  obstacle_stop when on, and uses the same stop_distance_m / stop_sector_deg /
  stop_confirm_ticks. If it cannot find a side it holds still ("give up") until
  /enable_inference is toggled. First field runs 2026-09-22
  (mission_carrot_sidestep*): 23 side-steps, all real obstacles, none hit --
  but in a saw-tooth along the same hedge, which is what
  checkpoint_controller_node's local_replan is for. controller_sim, carrot
  1.5 m, 8 seeds x 8 scenarios: alone 1 collision and 57/64 completed; with
  local_replan 0 collisions and 63/64.

The carrot distance is a parameter of checkpoint_controller_node, which the
mission launch leaves to be started by hand -- pass carrot_distance_m there.
"""
import math
import threading

import numpy as np
import pyproj
import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, NavSatFix
from std_msgs.msg import Bool, Float32, String

from erc_inference import sidestep as ss
from erc_inference.motion_control import (
    DEFAULT_PARAMS,
    MotionController,
    goal_bearing_from_offset,
    parameter_errors,
    robot_frame_offset,
)

GOAL_RESET_DISTANCE_M = 3.0
# SideStep thresholds that already exist as obstacle_stop parameters
SIDESTEP_SHARED = {'stop_distance_m': 'stop_distance_m',
                   'sector_deg': 'stop_sector_deg',
                   'confirm_ticks': 'stop_confirm_ticks'}
NULL_PLAN = [(0.0, 0.0, 0.0, 0.0)] * 8
LATCHED_QOS = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)


_TO_UTM = {}


def to_utm(lat, lon, zone_of=None):
    """(easting, northing) in the UTM zone of `zone_of` (default: this point).

    pyproj, not the `utm` package omnivla_edge_node uses: this node runs as a
    plain entry point under /usr/bin/python3, which has pyproj (3.6.1 -- it is
    what checkpoint_controller_node already relies on) and NOT utm, so importing
    utm here would kill the node on its first line. Same projection, same
    numbers. Both points are forced into ONE zone -- utm.from_latlon picks a
    zone per point, which would give a garbage delta across a zone boundary.
    """
    zlat, zlon = zone_of if zone_of is not None else (lat, lon)
    zone = int((zlon + 180.0) // 6.0) + 1
    epsg = (32600 if zlat >= 0 else 32700) + zone
    tr = _TO_UTM.get(epsg)
    if tr is None:
        tr = _TO_UTM[epsg] = pyproj.Transformer.from_crs(
            'EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    return tr.transform(lon, lat)


def ground_distance_m(lat1, lon1, lat2, lon2):
    k = math.radians(1.0) * 6378137.0
    east = (lon2 - lon1) * k * math.cos(math.radians(0.5 * (lat1 + lat2)))
    return math.hypot(east, (lat2 - lat1) * k)


class CarrotControllerNode(Node):
    def __init__(self):
        super().__init__('carrot_controller_node')
        self.declare_parameter('gps_topic', '/erc/gps/filtered')
        self.declare_parameter('compass_topic', '/erc/heading_deg')
        self.declare_parameter('goal_gps_topic', '/goal_gps')
        self.declare_parameter('enable_inference_topic', '/enable_inference')
        self.declare_parameter('cmd_vel_topic', '/omnivla/cmd_vel')
        self.declare_parameter('debug_topic', '/omnivla_debug')
        for name, default in DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)
        # obstacle_stop: off unless asked for, see module docstring
        self.declare_parameter('obstacle_stop', False)
        self.declare_parameter('free_space_topic', '/erc/free_space')
        # 1.2, not 0.8. The brake only bites after the loop delay: 1.3 s at
        # 0.3 m/s is 0.39 m travelled, plus ~0.2 m for the two confirm ticks,
        # plus the ramp -- ~0.67 m before the rover is stationary. In
        # controller_sim with obstacles, 0.8 m still hit the kerb 5/5 at 0.03 m;
        # 1.2 m hit nothing (0/5 in kerb, post and chicane), stopping 0.37-0.42 m
        # short. Scale it with max_linear_vel if that changes.
        self.declare_parameter('stop_distance_m', 1.2)
        self.declare_parameter('stop_sector_deg', 20.0)
        self.declare_parameter('stop_confirm_ticks', 2)
        self.declare_parameter('clear_confirm_ticks', 3)
        self.declare_parameter('free_space_timeout_s', 1.5)
        self.declare_parameter('stop_if_perception_stale', True)
        self.declare_parameter('obstacle_sidestep', False)
        for name, default in ss.DEFAULTS.items():
            if name not in SIDESTEP_SHARED:
                self.declare_parameter(f'sidestep.{name}', default)

        # controller.yaml's /** block sets steering_source: plan for the model
        # nodes, and it reaches this node too. Override it -- loudly, so nobody
        # reading the yaml wonders why it did not take.
        src = self.get_parameter('polar.steering_source').value
        if src != 'carrot':
            self.set_parameters([Parameter('polar.steering_source',
                                           Parameter.Type.STRING, 'carrot')])
            self.get_logger().info(
                f'polar.steering_source was "{src}" (controller.yaml); forced to '
                '"carrot" -- this node has no model to steer by.')
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.lock = threading.Lock()
        self.controller = MotionController(self._control_params())
        self.current_lat = self.current_lon = None
        self.current_compass_deg = None
        self.goal_lat = self.goal_lon = None
        self.enable_inference = False
        self._scan = None
        self._scan_t = None
        self._blocked_n = 0
        self._clear_n = 0
        self._stopped = False
        self.sidestep = ss.SideStep(**self._sidestep_params())

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.dbg_pub = self.create_publisher(String, self.get_parameter('debug_topic').value, 10)
        self.create_subscription(NavSatFix, self.get_parameter('gps_topic').value, self._on_gps, 10)
        self.create_subscription(Float32, self.get_parameter('compass_topic').value, self._on_compass, 10)
        self.create_subscription(NavSatFix, self.get_parameter('goal_gps_topic').value,
                                 self._on_goal, LATCHED_QOS)
        self.create_subscription(Bool, self.get_parameter('enable_inference_topic').value,
                                 self._on_enable, LATCHED_QOS)
        self.create_subscription(LaserScan, self.get_parameter('free_space_topic').value,
                                 self._on_scan, 10)
        self.create_timer(1.0 / float(self.get_parameter('tick_rate').value), self._tick)

        stop = self.get_parameter('obstacle_stop').value
        side = self.get_parameter('obstacle_sidestep').value
        self.get_logger().info(
            f'Carrot controller ready (no model) | obstacle_stop={stop} obstacle_sidestep={side}'
            + (f" at {self.get_parameter('stop_distance_m').value:.2f} m" if stop or side else ''))
        if side:
            self.get_logger().warn(
                'obstacle_sidestep is ON (replaces obstacle_stop). It has never run in the '
                'field: watch /omnivla_debug for "[side-step]"; a give-up holds until '
                '/enable_inference is toggled.')
        elif stop:
            self.get_logger().warn(
                'obstacle_stop is ON. It has never run in the field: expect '
                'false stops, and watch /omnivla_debug for "[obstacle-stop]".')

    # -- parameters -----------------------------------------------------
    def _control_params(self, overrides=None):
        p = {n: self.get_parameter(n).value for n in DEFAULT_PARAMS}
        p.update(overrides or {})
        return p

    def _sidestep_params(self):
        p = {k: self.get_parameter(f'sidestep.{k}').value
             for k in ss.DEFAULTS if k not in SIDESTEP_SHARED}
        p.update({k: self.get_parameter(n).value for k, n in SIDESTEP_SHARED.items()})
        return p

    def _on_set_parameters(self, changes):
        for c in changes:
            if c.name == 'polar.steering_source' and c.value != 'carrot':
                return SetParametersResult(
                    successful=False,
                    reason='this node has no model; only steering_source=carrot runs here. '
                           'Use omnivla_edge_node for plan/model.')
        errors = parameter_errors(self._control_params(
            {c.name: c.value for c in changes if c.name in DEFAULT_PARAMS}))
        if errors:
            return SetParametersResult(successful=False, reason='; '.join(errors))
        return SetParametersResult(successful=True)

    # -- inputs ---------------------------------------------------------
    def _on_gps(self, m):
        with self.lock:
            self.current_lat, self.current_lon = m.latitude, m.longitude

    def _on_compass(self, m):
        with self.lock:
            self.current_compass_deg = m.data

    def _on_goal(self, m):
        with self.lock:
            jumped = (self.goal_lat is None or ground_distance_m(
                self.goal_lat, self.goal_lon, m.latitude, m.longitude) > GOAL_RESET_DISTANCE_M)
            self.goal_lat, self.goal_lon = m.latitude, m.longitude
            if jumped:
                self.controller.reset()

    def _on_enable(self, m):
        with self.lock:
            if m.data != self.enable_inference:
                self.controller.reset()
                # also the way out of a side-step give-up: operator toggles enable
                self.sidestep = ss.SideStep(**self._sidestep_params())
            self.enable_inference = m.data

    def _on_scan(self, m):
        with self.lock:
            self._scan = m
            self._scan_t = self._now()

    # -- obstacle stop --------------------------------------------------
    def _ahead_m(self):
        """Closest reported range in the sector straight ahead, or None if stale."""
        if self._scan is None or self._scan_t is None:
            return None
        if self._now() - self._scan_t > float(self.get_parameter('free_space_timeout_s').value):
            return None
        s = self._scan
        half = math.radians(float(self.get_parameter('stop_sector_deg').value))
        r = np.asarray(s.ranges, float)
        a = s.angle_min + np.arange(len(r)) * s.angle_increment
        k = (np.abs(a) <= half) & np.isfinite(r)
        return float(r[k].min()) if k.any() else None

    def _fresh_scan(self):
        """(bearings in deg, left-positive; free in m) or None if stale."""
        if self._scan is None or self._scan_t is None:
            return None
        if self._now() - self._scan_t > float(self.get_parameter('free_space_timeout_s').value):
            return None
        s = self._scan
        r = np.asarray(s.ranges, float)
        a = np.degrees(s.angle_min + np.arange(len(r)) * s.angle_increment)
        r = np.where(np.isfinite(r), r, s.range_max)
        return a, r

    def _sidestep_gate(self, v, w, hdg, cur, bearing):
        """-> (v, w, note, resumed). SideStep's yaw is CCW; the compass is CW."""
        scan = self._fresh_scan()
        if scan is None:
            if self.get_parameter('stop_if_perception_stale').value:
                self.get_logger().warn('obstacle_sidestep: no fresh /erc/free_space -- holding',
                                       throttle_duration_sec=5)
                return 0.0, 0.0, ' [side-step] perception stale', False
            return v, w, ' [side-step] perception stale, ignored', False
        before = self.sidestep.state
        v, w, note, resumed = self.sidestep.step(
            self._now(), -math.radians(hdg), cur, scan[0], scan[1], v, w, bearing)
        if self.sidestep.state != before:
            self.get_logger().warn(f'obstacle_sidestep: {before} -> {self.sidestep.state} {note}')
        return v, w, (f' {note}' if note else ''), resumed

    def _obstacle_gate(self, v):
        """-> (stop?, note). Hysteresis both ways so one noisy frame cannot stop or release."""
        if not self.get_parameter('obstacle_stop').value or v <= 0.0:
            return False, ''
        ahead = self._ahead_m()
        if ahead is None:
            # perception silent: the protection asked for is not there. Say so,
            # and by default fail safe rather than drive on blind.
            if self.get_parameter('stop_if_perception_stale').value:
                self.get_logger().warn('obstacle_stop: no fresh /erc/free_space -- holding',
                                       throttle_duration_sec=5)
                return True, ' [obstacle-stop] perception stale'
            return False, ' [obstacle-stop] perception stale, ignored'
        near = ahead < float(self.get_parameter('stop_distance_m').value)
        self._blocked_n = self._blocked_n + 1 if near else 0
        self._clear_n = 0 if near else self._clear_n + 1
        if not self._stopped and self._blocked_n >= int(self.get_parameter('stop_confirm_ticks').value):
            self._stopped = True
            self.get_logger().warn(f'obstacle_stop: {ahead:.2f} m ahead -- stopping')
        elif self._stopped and self._clear_n >= int(self.get_parameter('clear_confirm_ticks').value):
            self._stopped = False
            self.get_logger().info(f'obstacle_stop: clear ({ahead:.2f} m) -- resuming')
        return self._stopped, f' [obstacle-stop] ahead={ahead:.2f}m stopped={self._stopped}'

    # -- control --------------------------------------------------------
    def _tick(self):
        with self.lock:
            ready = (self.enable_inference and self.current_lat is not None
                     and self.current_compass_deg is not None and self.goal_lat is not None)
            if not ready:
                self._publish(0.0, 0.0)
                self.controller.note_idle(self._now())
                return
            lat, lon, hdg = self.current_lat, self.current_lon, self.current_compass_deg
            glat, glon = self.goal_lat, self.goal_lon

        cur = to_utm(lat, lon)
        goal = to_utm(glat, glon, zone_of=(lat, lon))
        rel_x, rel_y = robot_frame_offset(goal[0] - cur[0], goal[1] - cur[1], hdg)
        radius = math.hypot(rel_x, rel_y)
        bearing = goal_bearing_from_offset(rel_x, rel_y)
        params = self._control_params({'polar.steering_source': 'carrot'})
        # No model, so a null 8-waypoint plan. Not an empty list: command()
        # indexes waypoints[waypoint_select] unconditionally to log the model's
        # alpha, and [] would raise on the first tick. steering_source=carrot
        # never reads the plan for steering.
        mode, v, w, detail = self.controller.command(
            self._now(), params, bearing, float(radius), NULL_PLAN, 4, True)

        if self.get_parameter('obstacle_sidestep').value:
            v, w, note, resumed = self._sidestep_gate(v, w, hdg, cur, bearing)
            if resumed:
                # its delay compensation recorded turns it never commanded
                self.controller.reset()
        else:
            stop, note = self._obstacle_gate(v)
            if stop:
                v, w = 0.0, 0.0
                self.controller.note_idle(self._now())
        msg = (f'[{mode}] goal_bearing={math.degrees(bearing):+.1f}deg goal_dist={radius:.2f}m | '
               f'{detail}{note} | linear_cmd={v:.4f} angular_cmd={w:.4f} | modality=4 source=carrot')
        self.get_logger().info(msg)
        self.dbg_pub.publish(String(data=msg))
        self._publish(v, w)

    def _publish(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.cmd_pub.publish(t)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def destroy_node(self):
        # On Ctrl-C Jazzy has already shut the context down and publishing
        # raises, so this final stop only goes out on a clean destroy. Nothing
        # in this repo times out a stale cmd_vel; whether the SDK does is unverified.
        if self.context.ok():
            self._publish(0.0, 0.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarrotControllerNode()
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
