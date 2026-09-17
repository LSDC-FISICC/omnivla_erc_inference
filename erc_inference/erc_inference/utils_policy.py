import os
import sys
import io
import matplotlib.pyplot as plt

# ROS
#from sensor_msgs.msg import Image

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

import clip
import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional

def imgmsg_to_rgb8(msg) -> np.ndarray:
    """Decode a sensor_msgs/Image into an HxWx3 uint8 RGB array.

    Avoids cv_bridge: its compiled cv_bridge_boost extension is built
    against ROS's system numpy (1.x) and segfaults when loaded in a
    process where torch has already pulled in this venv's numpy 2.x.
    Only the encodings the camera topics actually publish are supported.
    """
    channels = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f"Unsupported image encoding for imgmsg_to_rgb8: {msg.encoding!r}")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding == "bgr8":
        img = img[:, :, ::-1]
    elif msg.encoding == "mono8":
        img = np.repeat(img, 3, axis=2)
    return np.ascontiguousarray(img)


#model architecture
from erc_inference.model_omnivla_edge import OmniVLA_edge


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    model_type = config["model_type"]
    
    if config["model_type"] == "omnivla-edge":
        model = OmniVLA_edge(        
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )  
        text_encoder, preprocess = clip.load(config["clip_type"])    
        text_encoder.to(torch.float32)    
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if model_type == "omnivla-edge":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    
    return model, text_encoder, preprocess

def transform_images_PIL_mask(pil_imgs: List[PILImage.Image], mask) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            #transforms.ToTensor(),        
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        transf_img = transform_type(TF.to_tensor(pil_img*mask)/255.0) #/255.0
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

def transform_images_PIL(pil_imgs: List[PILImage.Image]) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        transf_img = transform_type(pil_img.copy())
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

def transform_images_map(pil_imgs: List[PILImage.Image]) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    image_size_small = (96, 96)
    
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        pil_img = pil_img.resize(image_size_small) 
        transf_img = transform_type(pil_img)          
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
