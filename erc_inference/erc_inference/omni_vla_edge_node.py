#!/usr/bin/env python3
# ===============================================================
# OmniVLA-edge ROS2 Node
# ===============================================================
#
# - Loads the OmniVLA-edge model ONCE at node startup (not per-frame).
# - Subscribes to a camera topic to keep a rolling context of frames.
# - Subscribes to GPS (NavSatFix) + heading (Float32, degrees) for the
#   robot's *current* pose (this must be live, it changes every tick).
# - Exposes the *goal* (target GPS, target compass, language prompt,
#   goal image, which modalities to use, which predicted waypoint to
#   use, velocity limits, ...) as ROS2 parameters. Because they are
#   parameters, you can change them at any time while the node is
#   running, e.g.:
#
#     ros2 param set /omnivla_edge_node goal_lat 37.8738930785863
#     ros2 param set /omnivla_edge_node goal_lon -122.26746181032362
#     ros2 param set /omnivla_edge_node lan_inst_prompt "blue trash bin"
#     ros2 param set /omnivla_edge_node use_lan_prompt true
#
# - Runs inference on a timer (tick_rate Hz) and publishes the result
#   as a geometry_msgs/Twist on /cmd_vel.
#
# Place this file in a ROS2 python package (alongside model_omnivla_edge.py
# and utils_policy.py, which must be importable), add it as an entry
# point / node, and launch it.
# ===============================================================

import math
import threading
from collections import deque

import numpy as np
import torch
import utm
import clip
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from erc_inference.utils_policy import (
    transform_images_map,
    load_model,
    transform_images_PIL_mask,
)

IMG_SIZE = (96, 96)
IMG_SIZE_CLIP = (224, 224)
METRIC_WAYPOINT_SPACING = 0.1
THRES_DIST = 30.0


def clip_angle(angle: float) -> float:
    """Wrap an angle (rad) to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class OmniVLAEdgeNode(Node):
    def __init__(self):
        super().__init__("omnivla_edge_node")
        self.bridge = CvBridge()
        self.lock = threading.RLock()

        # ---------------------------------------------------------
        # Static / startup-only parameters (model + topics)
        # ---------------------------------------------------------
        self.declare_parameter("model_checkpoint_path", "./omnivla-edge/omnivla-edge.pth")
        self.declare_parameter("context_size", 5)
        self.declare_parameter("obs_encoder", "efficientnet-b0")
        self.declare_parameter("encoding_size", 256)
        self.declare_parameter("obs_encoding_size", 1024)
        self.declare_parameter("goal_encoding_size", 1024)
        self.declare_parameter("late_fusion", False)
        self.declare_parameter("mha_num_attention_heads", 4)
        self.declare_parameter("mha_num_attention_layers", 4)
        self.declare_parameter("mha_ff_dim_factor", 4)
        self.declare_parameter("clip_type", "ViT-B/32")
        self.declare_parameter("len_traj_pred", 8)
        self.declare_parameter("learn_angle", True)

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("gps_topic", "/gps/fix")
        self.declare_parameter("compass_topic", "/compass")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("tick_rate", 3.0)

        # ---------------------------------------------------------
        # Runtime-tunable parameters (goal / inference request).
        # These can be changed at any point while the node is alive.
        # ---------------------------------------------------------
        self.declare_parameter("enable_inference", True)
        self.declare_parameter("goal_lat", 0.0)
        self.declare_parameter("goal_lon", 0.0)
        self.declare_parameter("goal_compass_deg", 0.0)
        self.declare_parameter("lan_inst_prompt", "")
        self.declare_parameter("goal_image_path", "")
        self.declare_parameter("use_pose_goal", False)
        self.declare_parameter("use_satellite", False)
        self.declare_parameter("use_image_goal", True)
        self.declare_parameter("use_lan_prompt", False)
        self.declare_parameter("waypoint_select", 4)
        self.declare_parameter("max_linear_vel", 0.3)
        self.declare_parameter("max_angular_vel", 0.3)

        # ---------------------------------------------------------
        # Load model ONCE
        # ---------------------------------------------------------
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Loading OmniVLA-edge on {self.device} ...")

        model_params = {
            "model_type": "omnivla-edge",
            "len_traj_pred": self.get_parameter("len_traj_pred").value,
            "learn_angle": self.get_parameter("learn_angle").value,
            "context_size": self.get_parameter("context_size").value,
            "obs_encoder": self.get_parameter("obs_encoder").value,
            "encoding_size": self.get_parameter("encoding_size").value,
            "obs_encoding_size": self.get_parameter("obs_encoding_size").value,
            "goal_encoding_size": self.get_parameter("goal_encoding_size").value,
            "late_fusion": self.get_parameter("late_fusion").value,
            "mha_num_attention_heads": self.get_parameter("mha_num_attention_heads").value,
            "mha_num_attention_layers": self.get_parameter("mha_num_attention_layers").value,
            "mha_ff_dim_factor": self.get_parameter("mha_ff_dim_factor").value,
            "clip_type": self.get_parameter("clip_type").value,
        }
        ckpt_path = self.get_parameter("model_checkpoint_path").value
        self.model, self.text_encoder, self.preprocess = load_model(ckpt_path, model_params, self.device)
        self.model = self.model.to(self.device).eval()
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.context_size = model_params["context_size"]
        self.get_logger().info("Model loaded.")

        # No mask (fisheye-specific masking disabled, matches sample script default)
        self.mask_96 = np.ones((96, 96, 3), dtype=np.float32)
        self.mask_224 = np.ones((224, 224, 3), dtype=np.float32)

        # ---------------------------------------------------------
        # Mutable state, protected by self.lock
        # ---------------------------------------------------------
        self.context_queue = deque(maxlen=self.context_size + 1)
        self.latest_frame_full = None  # for the 224px "cur_large_img"
        self.current_lat = None
        self.current_lon = None
        self.current_compass_deg = None

        self.goal_image_pil = PILImage.new("RGB", IMG_SIZE, color=(0, 0, 0))
        self._load_goal_image(self.get_parameter("goal_image_path").value)

        # ---------------------------------------------------------
        # Pub / Sub
        # ---------------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.create_subscription(Image, self.get_parameter("image_topic").value, self.image_callback, 10)
        self.create_subscription(NavSatFix, self.get_parameter("gps_topic").value, self.gps_callback, 10)
        self.create_subscription(Float32, self.get_parameter("compass_topic").value, self.compass_callback, 10)

        self.add_on_set_parameters_callback(self.on_parameters_set)

        tick_rate = self.get_parameter("tick_rate").value
        self.timer = self.create_timer(1.0 / tick_rate, self.timer_callback)

        self.get_logger().info("OmniVLA-edge node ready. Goal parameters can be updated at any time via ros2 param.")

    # ---------------------------------------------------------
    # Parameter updates (this is how you "send parameters anytime")
    # ---------------------------------------------------------
    def on_parameters_set(self, params):
        with self.lock:
            for p in params:
                if p.name == "goal_image_path" and p.type_ == Parameter.Type.STRING and p.value:
                    if not self._load_goal_image(p.value):
                        return SetParametersResult(successful=False, reason=f"Could not open image: {p.value}")
        return SetParametersResult(successful=True)

    def _load_goal_image(self, path: str) -> bool:
        if not path:
            return True
        try:
            self.goal_image_pil = PILImage.open(path).convert("RGB").resize(IMG_SIZE)
            self.get_logger().info(f"Goal image updated from {path}")
            return True
        except Exception as e:
            self.get_logger().warn(f"Failed to load goal image '{path}': {e}")
            return False

    # ---------------------------------------------------------
    # Sensor callbacks
    # ---------------------------------------------------------
    def image_callback(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        pil_img = PILImage.fromarray(cv_img)
        with self.lock:
            self.latest_frame_full = pil_img.resize(IMG_SIZE_CLIP)
            self.context_queue.append(pil_img.resize(IMG_SIZE))

    def gps_callback(self, msg: NavSatFix):
        with self.lock:
            self.current_lat = msg.latitude
            self.current_lon = msg.longitude

    def compass_callback(self, msg: Float32):
        with self.lock:
            self.current_compass_deg = msg.data

    # ---------------------------------------------------------
    # Helpers (ported from run_omnivla_edge.py)
    # ---------------------------------------------------------
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    @staticmethod
    def compute_modality_id(pose_goal, satellite, image_goal, lan_prompt):
        if pose_goal and satellite and image_goal and not lan_prompt:
            return 3
        elif not pose_goal and satellite and not image_goal and not lan_prompt:
            return 0
        elif pose_goal and not satellite and not image_goal and not lan_prompt:
            return 4
        elif pose_goal and satellite and not image_goal and not lan_prompt:
            return 1
        elif not pose_goal and satellite and image_goal and not lan_prompt:
            return 2
        elif pose_goal and not satellite and image_goal and not lan_prompt:
            return 5
        elif not pose_goal and not satellite and image_goal and not lan_prompt:
            return 6
        elif not pose_goal and not satellite and not image_goal and lan_prompt:
            return 7
        elif pose_goal and not satellite and not image_goal and lan_prompt:
            return 8
        elif not pose_goal and not satellite and image_goal and lan_prompt:
            return 9
        # Fallback: no modality selected -> treat as satellite-only
        return 0

    # ---------------------------------------------------------
    # Main inference tick
    # ---------------------------------------------------------
    def timer_callback(self):
        with self.lock:
            enabled = self.get_parameter("enable_inference").value
            ready = (
                enabled
                and len(self.context_queue) == self.context_size + 1
                and self.latest_frame_full is not None
                and self.current_lat is not None
                and self.current_lon is not None
                and self.current_compass_deg is not None
            )
            if not ready:
                self.publish_cmd(0.0, 0.0)
                return

            # Snapshot everything we need under the lock, then release it
            # before running the (potentially slow) forward pass.
            context_queue = list(self.context_queue)
            cur_large_pil = self.latest_frame_full
            current_lat = self.current_lat
            current_lon = self.current_lon
            current_compass_deg = self.current_compass_deg

            goal_lat = self.get_parameter("goal_lat").value
            goal_lon = self.get_parameter("goal_lon").value
            goal_compass_deg = self.get_parameter("goal_compass_deg").value
            lan_inst_prompt = self.get_parameter("lan_inst_prompt").value
            goal_image_pil = self.goal_image_pil

            use_pose_goal = self.get_parameter("use_pose_goal").value
            use_satellite = self.get_parameter("use_satellite").value
            use_image_goal = self.get_parameter("use_image_goal").value
            use_lan_prompt = self.get_parameter("use_lan_prompt").value

            waypoint_select = self.get_parameter("waypoint_select").value
            max_linear = self.get_parameter("max_linear_vel").value
            max_angular = self.get_parameter("max_angular_vel").value

        try:
            linear_vel, angular_vel = self.run_inference(
                context_queue,
                cur_large_pil,
                current_lat,
                current_lon,
                current_compass_deg,
                goal_lat,
                goal_lon,
                goal_compass_deg,
                lan_inst_prompt,
                goal_image_pil,
                use_pose_goal,
                use_satellite,
                use_image_goal,
                use_lan_prompt,
                waypoint_select,
                max_linear,
                max_angular,
            )
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            self.publish_cmd(0.0, 0.0)
            return

        self.publish_cmd(linear_vel, angular_vel)

    def run_inference(
        self,
        context_queue,
        cur_large_pil,
        current_lat,
        current_lon,
        current_compass_deg,
        goal_lat,
        goal_lon,
        goal_compass_deg,
        lan_inst_prompt,
        goal_image_pil,
        use_pose_goal,
        use_satellite,
        use_image_goal,
        use_lan_prompt,
        waypoint_select,
        max_linear,
        max_angular,
    ):
        device = self.device

        # --- Current pose ---
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass_deg) / 180.0 * math.pi

        # --- Goal pose (relative, local frame) ---
        goal_utm = utm.from_latlon(goal_lat, goal_lon)
        goal_compass = -float(goal_compass_deg) / 180.0 * math.pi

        delta_x, delta_y = self.calculate_relative_position(cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1])
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x ** 2 + relative_y ** 2)
        if radius > THRES_DIST:
            relative_x *= THRES_DIST / radius
            relative_y *= THRES_DIST / radius

        goal_pose_torch = torch.from_numpy(np.array([
            relative_y / METRIC_WAYPOINT_SPACING,
            -relative_x / METRIC_WAYPOINT_SPACING,
            np.cos(goal_compass - cur_compass),
            np.sin(goal_compass - cur_compass),
        ])).unsqueeze(0).float().to(device)

        # --- Observation context ---
        obs_images = transform_images_PIL_mask(context_queue, self.mask_96)
        obs_images = torch.split(obs_images.to(device), 3, dim=1)
        obs_image_cur = obs_images[-1].to(device)
        obs_images = torch.cat(obs_images, dim=1).to(device)

        cur_large_img = transform_images_PIL_mask(cur_large_pil, self.mask_224).to(device)

        # --- Satellite imagery (dummy black placeholder; wire up a real
        # satellite/aerial tile subscriber here if use_satellite is used
        # in your deployment) ---
        satellite_cur = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
        satellite_goal = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
        current_map_image = transform_images_map(satellite_cur)
        goal_map_image = transform_images_map(satellite_goal)
        map_images = torch.cat((current_map_image.to(device), goal_map_image.to(device), obs_image_cur), axis=1)

        # --- Language instruction ---
        lan_inst = lan_inst_prompt if (use_lan_prompt and lan_inst_prompt) else "xxxx"
        obj_inst_lan = clip.tokenize(lan_inst, truncate=True).to(device)

        # --- Egocentric goal image ---
        goal_image = transform_images_PIL_mask(goal_image_pil, self.mask_96).to(device)

        modality_id = self.compute_modality_id(use_pose_goal, use_satellite, use_image_goal, use_lan_prompt)
        modality_id_select = torch.tensor([modality_id]).to(device)

        bimg = goal_image.size(0)
        with torch.no_grad():
            feat_text_lan = self.text_encoder.encode_text(obj_inst_lan)
            predicted_actions, distances, mask_number = self.model(
                obs_images.repeat(bimg, 1, 1, 1),
                goal_pose_torch.repeat(bimg, 1),
                map_images.repeat(bimg, 1, 1, 1),
                goal_image,
                modality_id_select.repeat(bimg),
                feat_text_lan.repeat(bimg, 1),
                cur_large_img.repeat(bimg, 1, 1, 1),
            )

        waypoints = predicted_actions.float().cpu().numpy()
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= METRIC_WAYPOINT_SPACING
        dx, dy, hx, hy = chosen_waypoint

        # --- PD controller -> (linear, angular) ---
        EPS = 1e-8
        DT = 1.0 / self.get_parameter("tick_rate").value
        if abs(dx) < EPS and abs(dy) < EPS:
            linear_vel_value = 0.0
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif abs(dx) < EPS:
            linear_vel_value = 0.0
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT

        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)

        # --- Velocity limiting (preserves turning radius) ---
        maxv, maxw = max_linear, max_angular
        if abs(linear_vel_value) <= maxv:
            if abs(angular_vel_value) <= maxw:
                linear_vel_limit = linear_vel_value
                angular_vel_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_limit = maxw * np.sign(linear_vel_value) * abs(rd)
                angular_vel_limit = maxw * np.sign(angular_vel_value)
        else:
            if abs(angular_vel_value) <= 0.001:
                linear_vel_limit = maxv * np.sign(linear_vel_value)
                angular_vel_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if abs(rd) >= maxv / maxw:
                    linear_vel_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_limit = maxv * np.sign(angular_vel_value) / abs(rd)
                else:
                    linear_vel_limit = maxw * np.sign(linear_vel_value) * abs(rd)
                    angular_vel_limit = maxw * np.sign(angular_vel_value)

        return float(linear_vel_limit), float(angular_vel_limit)

    def publish_cmd(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_vel_pub.publish(msg)

    def destroy_node(self):
        self.publish_cmd(0.0, 0.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OmniVLAEdgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()