#!/usr/bin/env python3
# ===============================================================
# OmniVLA-original ROS2 Node
# ===============================================================
#
# Equivalent to omnivla_edge_node.py but for the original OmniVLA model.
# It keeps the same ROS topic-driven behavior and runtime flags as the
# edge node so you can swap models without changing the rest of the robot
# stack.
# ===============================================================

import math
import os
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch
import utm
from PIL import Image as PILImage
from torch.nn.utils.rnn import pad_sequence

# Must come before `transformers`: clip's import chain is what first pulls in
# torch._dynamo/triton. Importing transformers first leaves torch._dynamo in
# a state that segfaults when triton's native extension loads afterwards
# (reproduced on this Spark's aarch64 + torch 2.13.0+cu130 + triton 3.7.1).
from erc_inference.utils_policy import imgmsg_to_rgb8

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Bool, Float32, Int32, String

from erc_inference.motion_control import (
    DEFAULT_PARAMS,
    MotionController,
    goal_bearing_from_offset,
    parameter_errors,
    polar_gain_warnings,
    robot_frame_offset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OMNIVLA_ROOT = PROJECT_ROOT / "OmniVLA"
if str(OMNIVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(OMNIVLA_ROOT))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM

IMG_SIZE = (96, 96)
IMG_SIZE_CLIP = (224, 224)
# Scale between the model's goal/waypoint units and meters, same value and same
# reason as omnivla_edge_node: frodobots_dataset.py:606 normalizes both the
# goal_pose the model reads and the waypoints it predicts by 0.25. This node
# used 0.1, which fed goals 2.5x too large and shrank the waypoints 2.5x.
METRIC_WAYPOINT_SPACING = 0.25
THRES_DIST = 30.0
# A goal that moves further than this between two messages is a new target (a
# new leg, or a replan), not the carrot's normal advance of a few cm per tick.
GOAL_RESET_DISTANCE_M = 3.0

# The goal and the modality flags are latched state, not events, and
# checkpoint_controller_node publishes them TRANSIENT_LOCAL. A volatile
# subscriber here would miss them when this node starts mid-leg and silently
# drive on the defaults below (modality 0, satellite, no GPS goal). Same
# profile as omnivla_edge_node and checkpoint_controller_node; all three
# must agree.
LATCHED_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def ground_distance_m(lat1, lon1, lat2, lon2):
    """Equirectangular distance -- ample for telling a carrot step from a new leg."""
    k = math.radians(1.0) * 6378137.0
    east = (lon2 - lon1) * k * math.cos(math.radians(0.5 * (lat1 + lat2)))
    return math.hypot(east, (lat2 - lat1) * k)


def strip_ddp_prefix(state_dict):
    """Remove DDP prefixes like 'module.' from checkpoint keys."""
    if not isinstance(state_dict, dict):
        return state_dict
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


class OmniVLAOriginalNode(Node):
    def __init__(self):
        super().__init__("omnivla_original_node")
        self.lock = threading.RLock()

        # Parameters aligned with omnivla_edge_node.py
        # runs/omnivla-original--120000_chkpt, NOT OMNIVLA_ROOT/"omnivla-original": same step label,
        # different weights. This is the checkpoint probe_omnivla.py / probe_swap_lora_head.py
        # confirmed healthy (std=0.237) before the FrodoBots fine-tune collapsed at 160k+.
        self.declare_parameter(
            "model_checkpoint_path", str(OMNIVLA_ROOT / "runs" / "omnivla-original--120000_chkpt")
        )
        self.declare_parameter("model_checkpoint_step", 120000)
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

        self.declare_parameter("image_topic", "/erc/front_camera")
        # The filtered fix, not raw /erc/gps: it is what checkpoint_controller_node
        # projects the carrot against, and raw refreshes every ~1.2 s in 0.4 m jumps.
        self.declare_parameter("gps_topic", "/erc/gps/filtered")
        self.declare_parameter("compass_topic", "/erc/heading_deg")
        self.declare_parameter("cmd_vel_topic", "/omnivla/cmd_vel")
        # Control parameters (including tick_rate, max_linear_vel, max_angular_vel):
        # defaults in motion_control.DEFAULT_PARAMS, values and rationale in
        # config/controller.yaml. Same set the edge node declares, so both models
        # run the same control law and the comparison isolates the model.
        for name, default in DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)

        self.declare_parameter("debug_topic", "/omnivla_debug")
        self.declare_parameter("goal_image_topic", "/goal_img")
        self.declare_parameter("goal_gps_topic", "/goal_gps")
        self.declare_parameter("goal_compass_topic", "/goal_compass")
        self.declare_parameter("lan_prompt_topic", "/lan_prompt")
        self.declare_parameter("use_pose_goal_topic", "/use_pose_goal")
        self.declare_parameter("use_satellite_topic", "/use_satellite")
        self.declare_parameter("use_image_goal_topic", "/use_image_goal")
        self.declare_parameter("use_lan_prompt_topic", "/use_lan_prompt")
        self.declare_parameter("waypoint_select_topic", "/waypoint_select")
        self.declare_parameter("enable_inference_topic", "/enable_inference")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Loading OmniVLA original on {self.device} ...")
        self.vla, self.action_head, self.pose_projector, self.processor = self.load_model()
        self.vla = self.vla.to(self.device).eval()
        # run_inference casts actions_hidden_states/modality_id to bfloat16 and calls
        # predict_action() outside the autocast block that covers self.vla(...), so
        # action_head's own weights must already be bfloat16 -- unlike pose_projector,
        # which stays float32 because it only runs inside that autocast block. Matches
        # OmniVLA/inference/run_omnivla.py's init_module(..., to_bf16=True) for action_head.
        self.action_head = self.action_head.to(self.device).to(torch.bfloat16).eval()
        self.pose_projector = self.pose_projector.to(self.device).eval()
        # Without this, transform_datatype gets action_tokenizer=None and emits an
        # empty action_chunk_string, so the prompt's labels never carry the
        # ACTION_TOKEN_BEGIN_IDX-range tokens get_current_action_mask/
        # get_next_actions_mask look for -- both masks come back all-False and the
        # actions_hidden_states reshape collapses to a zero-sized last dim.
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.context_size = self.get_parameter("context_size").value
        self.num_patches = self.vla.vision_backbone.get_num_patches() * self.vla.vision_backbone.get_num_images_in_input() + 1
        self.get_logger().info("Original OmniVLA model loaded.")

        self.mask_96 = np.ones((96, 96, 3), dtype=np.float32)
        self.mask_224 = np.ones((224, 224, 3), dtype=np.float32)

        self.context_queue = deque(maxlen=self.context_size + 1)
        self.latest_frame_full = None
        self.current_lat = None
        self.current_lon = None
        self.current_compass_deg = None

        self.goal_image_pil = PILImage.new("RGB", IMG_SIZE, color=(0, 0, 0))
        # None, not 0.0: (0.0, 0.0) is a real coordinate in the Gulf of Guinea, so a
        # 0.0 default is indistinguishable from a goal that actually arrived and
        # would aim the rover at a bearing clamped to THRES_DIST, looking like a
        # plausible 30 m goal instead of failing loudly.
        self.goal_lat = None
        self.goal_lon = None
        self.goal_compass_deg = 0.0
        self.lan_inst_prompt = ""
        self.use_pose_goal = False
        self.use_satellite = True
        self.use_image_goal = False
        self.use_lan_prompt = False
        self.waypoint_select = 4
        self.enable_inference = True

        params = self._control_params()
        errors = parameter_errors(params)
        if errors:
            raise ValueError("; ".join(errors) + " (check config/controller.yaml)")
        for warning in polar_gain_warnings(params["polar.k_rho"], params["polar.k_alpha"], params["polar.k_beta"]):
            self.get_logger().warn(f"polar gains: {warning}")
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.controller = MotionController(params)
        self._last_controller_type = None

        self.cmd_vel_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.debug_pub = self.create_publisher(String, self.get_parameter("debug_topic").value, 10)

        self.create_subscription(Image, self.get_parameter("image_topic").value, self.image_callback, 10)
        self.create_subscription(NavSatFix, self.get_parameter("gps_topic").value, self.gps_callback, 10)
        self.create_subscription(Float32, self.get_parameter("compass_topic").value, self.compass_callback, 10)

        self.create_subscription(Image, self.get_parameter("goal_image_topic").value, self.goal_image_callback, 10)
        self.create_subscription(NavSatFix, self.get_parameter("goal_gps_topic").value, self.goal_gps_callback, LATCHED_QOS)
        self.create_subscription(Float32, self.get_parameter("goal_compass_topic").value, self.goal_compass_callback, LATCHED_QOS)
        self.create_subscription(String, self.get_parameter("lan_prompt_topic").value, self.lan_prompt_callback, 10)
        self.create_subscription(Bool, self.get_parameter("use_pose_goal_topic").value, self.use_pose_goal_callback, LATCHED_QOS)
        self.create_subscription(Bool, self.get_parameter("use_satellite_topic").value, self.use_satellite_callback, LATCHED_QOS)
        self.create_subscription(Bool, self.get_parameter("use_image_goal_topic").value, self.use_image_goal_callback, LATCHED_QOS)
        self.create_subscription(Bool, self.get_parameter("use_lan_prompt_topic").value, self.use_lan_prompt_callback, LATCHED_QOS)
        self.create_subscription(Int32, self.get_parameter("waypoint_select_topic").value, self.waypoint_select_callback, 10)
        self.create_subscription(Bool, self.get_parameter("enable_inference_topic").value, self.enable_inference_callback, LATCHED_QOS)

        tick_rate = self.get_parameter("tick_rate").value
        self.timer = self.create_timer(1.0 / tick_rate, self.timer_callback)

        self.get_logger().info(
            "OmniVLA original node ready. Publish to the goal topics to update the inference request."
        )

    def load_model(self):
        model_dir = self.get_parameter("model_checkpoint_path").value
        step = self.get_parameter("model_checkpoint_step").value

        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"OmniVLA model directory not found: {model_dir}")

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        vla.vision_backbone.set_num_images_in_input(2)

        pose_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=POSE_DIM)
        action_head = L1RegressionActionHead_idcat(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=ACTION_DIM,
        )

        action_path = os.path.join(model_dir, f"action_head--{step}_checkpoint.pt")
        proprio_path = os.path.join(model_dir, f"proprio_projector--{step}_checkpoint.pt")
        if not os.path.exists(proprio_path):
            proprio_path = os.path.join(model_dir, f"pose_projector--{step}_checkpoint.pt")

        # Both heads are randomly initialised above. Loading them was previously
        # best-effort, so a model_checkpoint_step that does not match the files in
        # model_checkpoint_path left the node driving on random weights without
        # saying so -- and the two are set independently.
        for path, module, what in ((action_path, action_head, "action head"),
                                   (proprio_path, pose_projector, "proprio projector")):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{what} checkpoint not found: {path} -- check that "
                    f"model_checkpoint_step ({step}) matches model_checkpoint_path ({model_dir})")
            state_dict = torch.load(path, map_location=self.device)
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            module.load_state_dict(strip_ddp_prefix(state_dict), strict=True)
            self.get_logger().info(f"Loaded {what} checkpoint: {path}")

        return vla, action_head, pose_projector, processor

    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

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
        return 0

    def image_callback(self, msg: Image):
        cv_img = imgmsg_to_rgb8(msg)
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

    def goal_image_callback(self, msg: Image):
        cv_img = imgmsg_to_rgb8(msg)
        pil_img = PILImage.fromarray(cv_img).resize(IMG_SIZE)
        with self.lock:
            self.goal_image_pil = pil_img
        self.get_logger().info("Goal image updated.")

    def goal_gps_callback(self, msg: NavSatFix):
        with self.lock:
            # The carrot advances a few cm per tick; only a new leg or a replan
            # moves it by meters. Reset the controller on those, not on every step.
            jumped = (self.goal_lat is None or ground_distance_m(
                self.goal_lat, self.goal_lon, msg.latitude, msg.longitude) > GOAL_RESET_DISTANCE_M)
            self.goal_lat = msg.latitude
            self.goal_lon = msg.longitude
            if jumped:
                self.controller.reset()
        if jumped:
            self.get_logger().info(f"New goal: lat={msg.latitude}, lon={msg.longitude}")

    def goal_compass_callback(self, msg: Float32):
        with self.lock:
            self.goal_compass_deg = msg.data

    def lan_prompt_callback(self, msg: String):
        with self.lock:
            self.lan_inst_prompt = msg.data
        self.get_logger().info(f"Language prompt updated: '{msg.data}'")

    def use_pose_goal_callback(self, msg: Bool):
        with self.lock:
            self.use_pose_goal = msg.data

    def use_satellite_callback(self, msg: Bool):
        with self.lock:
            self.use_satellite = msg.data

    def use_image_goal_callback(self, msg: Bool):
        with self.lock:
            self.use_image_goal = msg.data

    def use_lan_prompt_callback(self, msg: Bool):
        with self.lock:
            self.use_lan_prompt = msg.data

    def waypoint_select_callback(self, msg: Int32):
        with self.lock:
            self.waypoint_select = msg.data

    def enable_inference_callback(self, msg: Bool):
        with self.lock:
            was_enabled = self.enable_inference
            self.enable_inference = msg.data
            # The rover is stopped while disabled (checkpoint verification, and the
            # gaps between legs). Resuming with the state from before the stop
            # would kick on the first tick.
            if was_enabled != msg.data:
                self.controller.reset()
        self.get_logger().info(f"enable_inference set to {msg.data}")

    def timer_callback(self):
        with self.lock:
            # A pose goal is only required by the modalities that actually read the
            # goal_pose token (use_pose_goal); the others have it masked inside the
            # model, so gating them on a GPS goal would stall them for no reason.
            have_pose_goal = self.goal_lat is not None and self.goal_lon is not None
            ready = (
                self.enable_inference
                and len(self.context_queue) == self.context_size + 1
                and self.latest_frame_full is not None
                and self.current_lon is not None
                and self.current_compass_deg is not None
                and (have_pose_goal or not self.use_pose_goal)
            )
            if not ready:
                self.publish_cmd(0.0, 0.0)
                self.controller.note_idle(self._now_s())
                if self.enable_inference:
                    self.get_logger().info(
                        f"enable_inference={self.enable_inference}, context_queue={len(self.context_queue)}/{self.context_size + 1}, "
                        f"latest_frame_full={self.latest_frame_full is not None}, current_lat={self.current_lat is not None}, "
                        f"current_lon={self.current_lon is not None}, current_compass_deg={self.current_compass_deg is not None}, "
                        f"goal_gps={have_pose_goal} (required={self.use_pose_goal})"
                    )
                return

            context_queue = list(self.context_queue)
            cur_large_pil = self.latest_frame_full
            current_lat = self.current_lat
            current_lon = self.current_lon
            current_compass_deg = self.current_compass_deg

            # Reachable with no goal only when use_pose_goal is False, i.e. the model
            # masks this token anyway. Fall back to the current position so the goal
            # vector is (0, 0) instead of feeding None into UTM.
            goal_lat = self.goal_lat if self.goal_lat is not None else self.current_lat
            goal_lon = self.goal_lon if self.goal_lon is not None else self.current_lon
            goal_compass_deg = self.goal_compass_deg
            lan_inst_prompt = self.lan_inst_prompt
            goal_image_pil = self.goal_image_pil

            use_pose_goal = self.use_pose_goal
            use_satellite = self.use_satellite
            use_image_goal = self.use_image_goal
            use_lan_prompt = self.use_lan_prompt

            waypoint_select = self.waypoint_select

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
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self.get_logger().error(f"Inference failed: {exc}")
            self.publish_cmd(0.0, 0.0)
            self.controller.note_idle(self._now_s())
            return

        self.publish_cmd(linear_vel, angular_vel)

    def data_transformer_omnivla(self, current_image_pil, lan_inst, goal_image_pil, goal_pose_loc_norm):
        actions = np.random.rand(8, 4)
        goal_pose_cos_sin = goal_pose_loc_norm
        batch_data = self.transform_datatype(
            lan_inst,
            actions,
            goal_pose_cos_sin,
            current_image_pil,
            goal_image_pil,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            base_tokenizer=self.processor.tokenizer,
            image_transform=self.processor.image_processor.apply_transform,
        )
        batch = self.collator_custom(
            instances=[batch_data],
            model_max_length=self.processor.tokenizer.model_max_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        return batch

    @staticmethod
    def collator_custom(instances, model_max_length, pad_token_id, padding_side="right"):
        ignore_index = -100
        input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=ignore_index)
        input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
            pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )
        return output

    def transform_datatype(
        self,
        inst_obj,
        actions,
        goal_pose_cos_sin,
        current_image_pil,
        goal_image_pil,
        prompt_builder,
        action_tokenizer,
        base_tokenizer,
        image_transform,
        predict_stop_token=True,
    ):
        ignore_index = -100
        current_action = actions[0]
        future_actions = actions[1:]
        if action_tokenizer is not None:
            future_actions_string = "".join(action_tokenizer(future_actions))
            current_action_string = action_tokenizer(current_action)
            action_chunk_string = current_action_string + future_actions_string
        else:
            action_chunk_string = ""

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        prompt_builder = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(len(action_chunk_string) + 1)] = ignore_index
        if not predict_stop_token:
            labels[-1] = ignore_index

        pixel_values_current = image_transform(current_image_pil)
        pixel_values_goal = image_transform(goal_image_pil)

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            actions=torch.as_tensor(actions),
            goal_pose=goal_pose_cos_sin,
            img_pil=current_image_pil,
            inst=inst_obj,
        )

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
    ):
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass_deg) / 180.0 * math.pi

        goal_utm = utm.from_latlon(goal_lat, goal_lon)
        goal_compass = -float(goal_compass_deg) / 180.0 * math.pi

        delta_x, delta_y = self.calculate_relative_position(cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1])
        relative_x, relative_y = robot_frame_offset(delta_x, delta_y, current_compass_deg)
        radius = np.sqrt(relative_x ** 2 + relative_y ** 2)
        # Where the goal really is, from localization alone, before the clamp below.
        # GoalTurn acts on this, since the model's waypoint cannot point behind.
        goal_bearing = goal_bearing_from_offset(relative_x, relative_y)
        if radius > THRES_DIST:
            relative_x *= THRES_DIST / radius
            relative_y *= THRES_DIST / radius

        goal_pose_loc_norm = np.array([
            relative_y / METRIC_WAYPOINT_SPACING,
            -relative_x / METRIC_WAYPOINT_SPACING,
            np.cos(goal_compass - cur_compass),
            np.sin(goal_compass - cur_compass),
        ]).astype(np.float32)

        current_image_pil = context_queue[-1] if context_queue else cur_large_pil
        lan_inst = lan_inst_prompt if (use_lan_prompt and lan_inst_prompt) else "xxxx"
        modality_id_value = self.compute_modality_id(use_pose_goal, use_satellite, use_image_goal, use_lan_prompt)
        modality_id = torch.tensor([modality_id_value], dtype=torch.float32, device=self.device)

        batch = self.data_transformer_omnivla(current_image_pil, lan_inst, goal_image_pil, goal_pose_loc_norm)
        batch["goal_pose"] = torch.as_tensor(goal_pose_loc_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        batch["input_ids"] = batch["input_ids"].to(self.device)
        batch["attention_mask"] = batch["attention_mask"].to(self.device)
        batch["labels"] = batch["labels"].to(self.device)
        batch["pixel_values"] = batch["pixel_values"].to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = self.vla(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"].to(torch.bfloat16),
                        modality_id=modality_id.to(torch.bfloat16),
                        labels=batch["labels"],
                        output_hidden_states=True,
                        proprio=batch["goal_pose"].to(torch.bfloat16),
                        proprio_projector=self.pose_projector,
                        use_film=False,
                    )
            else:
                output = self.vla(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    modality_id=modality_id,
                    labels=batch["labels"],
                    output_hidden_states=True,
                    proprio=batch["goal_pose"],
                    proprio_projector=self.pose_projector,
                    use_film=False,
                )

        ground_truth_token_ids = batch["labels"][:, 1:].to(self.device)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, self.num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            predicted_actions = self.action_head.predict_action(
                actions_hidden_states,
                modality_id.to(torch.bfloat16),
            )

        waypoints = predicted_actions.float().cpu().numpy()
        chunk = waypoints[0].copy()
        chunk[:, :2] *= METRIC_WAYPOINT_SPACING
        dx, dy, hx, hy = chunk[waypoint_select]

        self.get_logger().info(
            f"[modality={modality_id_value}] raw_waypoint(dx={dx:.3f}, dy={dy:.3f}, hx={hx:.3f}, hy={hy:.3f}) "
            f"waypoint_select={waypoint_select} all_waypoints_shape={waypoints.shape}"
        )

        # --- Controller -> (linear, angular) ---
        params = self._control_params()
        if params["controller_type"] != self._last_controller_type:
            self.get_logger().info(f"Motion controller: {params['controller_type']}")
            self._last_controller_type = params["controller_type"]
        mode, linear_cmd, angular_cmd, detail = self.controller.command(
            self._now_s(), params, goal_bearing, float(radius), chunk.tolist(),
            waypoint_select, use_pose_goal)

        debug_msg = (
            f"[{mode}] goal_bearing={math.degrees(goal_bearing):+.1f}deg goal_dist={radius:.2f}m | "
            f"{detail} | linear_cmd={linear_cmd:.4f} angular_cmd={angular_cmd:.4f} | "
            f"modality={modality_id_value} lan_prompt='{lan_inst}'"
        )
        self.get_logger().info(debug_msg)
        self.debug_pub.publish(String(data=debug_msg))

        return float(linear_cmd), float(angular_cmd)

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _control_params(self, overrides=None):
        params = {name: self.get_parameter(name).value for name in DEFAULT_PARAMS}
        params.update(overrides or {})
        return params

    def _on_set_parameters(self, changes):
        """Reject parameter sets the controller cannot run; warn on unstable polar gains."""
        params = self._control_params({p.name: p.value for p in changes if p.name in DEFAULT_PARAMS})
        errors = parameter_errors(params)
        if errors:
            return SetParametersResult(successful=False, reason="; ".join(errors))
        if any(p.name in ("polar.k_rho", "polar.k_alpha", "polar.k_beta") for p in changes):
            for warning in polar_gain_warnings(params["polar.k_rho"], params["polar.k_alpha"], params["polar.k_beta"]):
                self.get_logger().warn(f"polar gains: {warning}")
        return SetParametersResult(successful=True)

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
    node = OmniVLAOriginalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
