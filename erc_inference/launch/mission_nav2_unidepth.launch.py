"""mission_nav2.launch.py (Nav2 MPPI) with UniDepthV2 as the free-space depth model.

Same A* route, Nav2 MPPI and route follower; the only change is
the depth model behind /erc/free_space: depth_backend:=unidepth. Why: DA3's
relative depth compresses what is far, and in Panama's parks a tree line 20 m
away read as a wall at 0.4-1.1 m in 53% of the frames driven; UniDepthV2 does
not (20%, offline over mission_dry_run; every real obstacle of the 22/24-sept
Wuhan bags but two borderline ones). See erc_perception free_space_node.

Every mission_nav2 argument works here and means the same, e.g.

    ros2 launch erc_inference mission_nav2_unidepth.launch.py

Needs ~/lsdc/erc-omni-vla/.venv-unidepth (UniDepth installed, torch shared from
.venv-depth); the model weights are cached, no internet needed.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg='mission_nav2 with depth_backend:=unidepth (UniDepthV2)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('erc_inference'), 'launch', 'mission_nav2.launch.py')),
            launch_arguments={'depth_backend': 'unidepth'}.items()),
    ])
