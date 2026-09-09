#!/usr/bin/env python3
# ===============================================================
# OmniVLA-edge test publisher
# ===============================================================
#
# Standalone node to exercise omnivla_edge_node.py WITHOUT a real
# robot. It publishes:
#
#   - /camera/image_raw  (continuously, simulates the live camera feed
#                          that fills the node's context_queue)
#   - /gps/fix            (continuously, current robot GPS)
#   - /compass             (continuously, current robot heading)
#   - /goal_img, /goal_gps, /goal_compass, /lan_prompt,
#     /use_pose_goal, /use_satellite, /use_image_goal, /use_lan_prompt,
#     /waypoint_select, /enable_inference
#                          (republished periodically so they arrive even
#                           if this script starts before omnivla_edge_node,
#                           and so you can watch the pipeline run end-to-end)
#
# You can pass real image files (--image / --goal-image), or leave them
# out and it will generate synthetic images (a moving gradient for the
# "camera" feed, a solid color square for the goal) so you can smoke-test
# the node without any real photos.
#
# Usage examples:
#
#   # Pure synthetic smoke test, defaults for everything
#   ros2 run <your_pkg> omnivla_test_publisher.py
#
#   # With real images and a language goal
#   python3 omnivla_test_publisher.py \
#       --image ./sample_scene.jpg \
#       --goal-image ./sample_goal.jpg \
#       --lan-prompt "blue trash bin" \
#       --use-lan-prompt
#
#   # GPS-goal style test
#   python3 omnivla_test_publisher.py \
#       --current-lat 37.8715 --current-lon -122.2730 --current-compass-deg 90 \
#       --goal-lat 37.8739 --goal-lon -122.2675 --goal-compass-deg 0 \
#       --use-pose-goal
#
# Stop with Ctrl+C. It will keep publishing the live sensor topics at
# --rate Hz and re-publish the goal topics every --goal-period seconds
# so you can watch /cmd_vel and /omnivla_debug update on the node side.
# ===============================================================

import argparse
import time

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Float32, String, Bool, Int32
from cv_bridge import CvBridge

CAM_SIZE = (224, 224)   # published as full-res; omnivla_edge_node resizes as needed
GOAL_SIZE = (224, 224)


def synthetic_camera_frame(t: float) -> np.ndarray:
    """A moving gradient so consecutive frames aren't identical (context
    queue diversity matters for a meaningful smoke test)."""
    h, w = CAM_SIZE[1], CAM_SIZE[0]
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    shift = (t * 0.3) % 1.0
    r = ((xx + shift) % 1.0 * 255).astype(np.uint8)
    g = (yy * 255).astype(np.uint8)
    b = np.full((h, w), 128, dtype=np.uint8)
    return np.stack([r, g, b], axis=-1)


def synthetic_goal_frame() -> np.ndarray:
    """A flat colored square, stands out from the gradient camera feed."""
    h, w = GOAL_SIZE[1], GOAL_SIZE[0]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (30, 180, 90)  # green-ish "goal marker"
    return img


class OmniVLATestPublisher(Node):
    def __init__(self, args):
        super().__init__("omnivla_test_publisher")
        self.args = args
        self.bridge = CvBridge()
        self.t0 = time.time()

        # --- Live sensor publishers ---
        self.image_pub = self.create_publisher(Image, args.image_topic, 10)
        self.gps_pub = self.create_publisher(NavSatFix, args.gps_topic, 10)
        self.compass_pub = self.create_publisher(Float32, args.compass_topic, 10)

        # --- Goal / inference-request publishers ---
        self.goal_image_pub = self.create_publisher(Image, args.goal_image_topic, 10)
        self.goal_gps_pub = self.create_publisher(NavSatFix, args.goal_gps_topic, 10)
        self.goal_compass_pub = self.create_publisher(Float32, args.goal_compass_topic, 10)
        self.lan_prompt_pub = self.create_publisher(String, args.lan_prompt_topic, 10)
        self.use_pose_goal_pub = self.create_publisher(Bool, args.use_pose_goal_topic, 10)
        self.use_satellite_pub = self.create_publisher(Bool, args.use_satellite_topic, 10)
        self.use_image_goal_pub = self.create_publisher(Bool, args.use_image_goal_topic, 10)
        self.use_lan_prompt_pub = self.create_publisher(Bool, args.use_lan_prompt_topic, 10)
        self.waypoint_select_pub = self.create_publisher(Int32, args.waypoint_select_topic, 10)
        self.enable_inference_pub = self.create_publisher(Bool, args.enable_inference_topic, 10)

        # --- Pre-load / pre-generate images once ---
        self.camera_img_static = self._load_image(args.image) if args.image else None
        self.goal_img_static = self._load_image(args.goal_image) if args.goal_image else None

        # --- Timers ---
        self.create_timer(1.0 / args.rate, self.publish_live_sensors)
        self.create_timer(args.goal_period, self.publish_goal_state)

        # Publish goal state once immediately too (in addition to the
        # periodic republish), so a node that's already up gets it fast.
        self.publish_goal_state()

        self.get_logger().info(
            f"Publishing live sensors @ {args.rate} Hz on "
            f"{args.image_topic}, {args.gps_topic}, {args.compass_topic}; "
            f"republishing goal state every {args.goal_period}s."
        )

    def _load_image(self, path: str) -> np.ndarray:
        try:
            img = PILImage.open(path).convert("RGB")
            return np.array(img)
        except Exception as e:
            self.get_logger().warn(f"Could not load '{path}' ({e}), falling back to synthetic image.")
            return None

    def _to_image_msg(self, arr: np.ndarray) -> Image:
        return self.bridge.cv2_to_imgmsg(arr, encoding="rgb8")

    # ---------------------------------------------------------
    # Continuous "live robot state" publishing
    # ---------------------------------------------------------
    def publish_live_sensors(self):
        t = time.time() - self.t0

        frame = self.camera_img_static if self.camera_img_static is not None else synthetic_camera_frame(t)
        self.image_pub.publish(self._to_image_msg(frame))

        gps_msg = NavSatFix()
        gps_msg.latitude = self.args.current_lat
        gps_msg.longitude = self.args.current_lon
        self.gps_pub.publish(gps_msg)

        compass_msg = Float32()
        compass_msg.data = self.args.current_compass_deg
        self.compass_pub.publish(compass_msg)

    # ---------------------------------------------------------
    # Periodic "goal / inference request" publishing
    # ---------------------------------------------------------
    def publish_goal_state(self):
        goal_frame = self.goal_img_static if self.goal_img_static is not None else synthetic_goal_frame()
        self.goal_image_pub.publish(self._to_image_msg(goal_frame))

        goal_gps_msg = NavSatFix()
        goal_gps_msg.latitude = self.args.goal_lat
        goal_gps_msg.longitude = self.args.goal_lon
        self.goal_gps_pub.publish(goal_gps_msg)

        goal_compass_msg = Float32()
        goal_compass_msg.data = self.args.goal_compass_deg
        self.goal_compass_pub.publish(goal_compass_msg)

        self.lan_prompt_pub.publish(String(data=self.args.lan_prompt))

        self.use_pose_goal_pub.publish(Bool(data=self.args.use_pose_goal))
        self.use_satellite_pub.publish(Bool(data=self.args.use_satellite))
        self.use_image_goal_pub.publish(Bool(data=self.args.use_image_goal))
        self.use_lan_prompt_pub.publish(Bool(data=self.args.use_lan_prompt))

        self.waypoint_select_pub.publish(Int32(data=self.args.waypoint_select))
        self.enable_inference_pub.publish(Bool(data=not self.args.start_disabled))

        self.get_logger().info(
            f"Goal republished: lat={self.args.goal_lat}, lon={self.args.goal_lon}, "
            f"compass={self.args.goal_compass_deg}, prompt='{self.args.lan_prompt}', "
            f"modalities(pose={self.args.use_pose_goal}, sat={self.args.use_satellite}, "
            f"img={self.args.use_image_goal}, lan={self.args.use_lan_prompt}), "
            f"waypoint_select={self.args.waypoint_select}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="OmniVLA-edge test publisher")

    # Topic names (should match omnivla_edge_node's topic parameters)
    p.add_argument("--image-topic", default="/camera/image_raw")
    p.add_argument("--gps-topic", default="/gps/fix")
    p.add_argument("--compass-topic", default="/compass")
    p.add_argument("--goal-image-topic", default="/goal_img")
    p.add_argument("--goal-gps-topic", default="/goal_gps")
    p.add_argument("--goal-compass-topic", default="/goal_compass")
    p.add_argument("--lan-prompt-topic", default="/lan_prompt")
    p.add_argument("--use-pose-goal-topic", default="/use_pose_goal")
    p.add_argument("--use-satellite-topic", default="/use_satellite")
    p.add_argument("--use-image-goal-topic", default="/use_image_goal")
    p.add_argument("--use-lan-prompt-topic", default="/use_lan_prompt")
    p.add_argument("--waypoint-select-topic", default="/waypoint_select")
    p.add_argument("--enable-inference-topic", default="/enable_inference")

    # Optional real image files; falls back to synthetic if omitted
    p.add_argument("--image", default=None, help="Path to an image used as the live camera frame")
    p.add_argument("--goal-image", default=None, help="Path to an image used as the goal image")

    # Current robot pose (fake GPS/compass)
    p.add_argument("--current-lat", type=float, default=37.8715)
    p.add_argument("--current-lon", type=float, default=-122.2730)
    p.add_argument("--current-compass-deg", type=float, default=0.0)

    # Goal pose / request
    p.add_argument("--goal-lat", type=float, default=37.8739)
    p.add_argument("--goal-lon", type=float, default=-122.2675)
    p.add_argument("--goal-compass-deg", type=float, default=0.0)
    p.add_argument("--lan-prompt", default="")
    p.add_argument("--use-pose-goal", action="store_true")
    p.add_argument("--use-satellite", action="store_true")
    p.add_argument("--use-image-goal", action="store_true")
    p.add_argument("--use-lan-prompt", action="store_true")
    p.add_argument("--waypoint-select", type=int, default=4)
    p.add_argument("--start-disabled", action="store_true", help="Publish enable_inference=false initially")

    # Timing
    p.add_argument("--rate", type=float, default=5.0, help="Hz for live camera/gps/compass publishing")
    p.add_argument("--goal-period", type=float, default=2.0, help="Seconds between goal-state republishes")

    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = OmniVLATestPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()