"""mission.launch.py, but with OmniVLA-original as the policy instead of the edge model.

Identical stack and identical ordering -- the only difference is which wrapper
runs as the inference layer:

    mission.launch.py           omni_vla_wrapper           -> omnivla_edge_node
    mission_omnivla.launch.py   omni_vla_original_wrapper  -> omnivla_original_node

Both wrappers load config/controller.yaml, so the motion controller (polar law,
carrot, GoalTurn, delay compensation, the 0.25 m/s floor) is the same on either
side and a comparison between the two isolates the model. Everything downstream
is untouched: both nodes publish /omnivla/cmd_vel and checkpoint_controller_node
applies the same acceleration limits on the way to /cmd_vel.

The same two things stay out of this launch, for the same reason as in
mission.launch.py -- the controller is what an operator kills to take the rover
back:

    ros2 run erc_inference checkpoint_controller_node
    ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \
        "{resume_from_latest_scanned: false}"

LIVE field operation only; it starts the SDK bridge.

The original model is ~7B and loads ~15 GB of weights, so the inference layer
takes considerably longer to come up than it does with the edge model. It is
started first here (as in mission.launch.py) and the rest of the stack does not
wait for it, but the rover will not move until the node reports ready.
"""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# See mission.launch.py: localization_global.launch.py blocks on the first live
# /erc/gps fix inside an OpaqueFunction, which stalls the launch's own asyncio
# loop -- including spawning the bridge that publishes that fix. Deferring the
# include behind a timer lets the bridge start polling first.
LOCALIZATION_DELAY_S = 5.0


def generate_launch_description():
    # Loads config/controller.yaml and fails loudly if it is missing, which
    # omni_vla_original_wrapper does the same way omni_vla_wrapper does. Run
    # through a wrapper rather than as a Node because the model needs its conda
    # environment (or the venv fallback on the Spark) and OmniVLA on PYTHONPATH,
    # which a plain exec of the entry point does not get.
    omnivla_wrapper = os.path.join(
        get_package_prefix('erc_inference'), 'lib', 'erc_inference', 'omni_vla_original_wrapper')

    bridge = [
        Node(package='erc_bridge', executable='erc_data_node',
             name='erc_data_node', output='screen'),
        Node(package='erc_bridge', executable='erc_screenshot_node',
             name='erc_screenshot_node', output='screen'),
        Node(package='erc_bridge', executable='erc_control_node',
             name='erc_control_node', output='screen'),
        # Read-only: polls GET /status and publishes /erc/telemetry_age and
        # /erc/frame_age_front|rear so they land in the bag. SDK v6.3 reports
        # the age of the telemetry and of the newest cached camera frame, which
        # is what separates the two readings ARQUITECTURA_ACTUAL Sec 8.4 leaves
        # open (real transport delay vs a rover clock ~1.1 s behind) and gives
        # the camera content lag TAREA1 Sec 6.2 had to infer from a fit.
        # mission_wuhan_hard was recorded without it and both stay unresolved.
        # Nothing consumes these topics; none of them reach control.
        Node(package='erc_bridge', executable='erc_status_node',
             name='erc_status_node', output='screen'),
    ]

    static_map = [
        Node(package='erc_static_map', executable='erc_static_map_node',
             name='erc_static_map_node', output='screen'),
        Node(package='erc_static_map', executable='erc_astar_planner_node',
             name='erc_astar_planner_node', output='screen'),
    ]

    # Started early on purpose: loading the checkpoint takes far longer than
    # anything else here, and it has nothing to wait for.
    inference = ExecuteProcess(cmd=[omnivla_wrapper], output='screen')

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('erc_localization'), 'launch',
                         'localization_global.launch.py')))

    return LaunchDescription(
        bridge + static_map + [
            inference,
            TimerAction(period=LOCALIZATION_DELAY_S, actions=[localization]),
        ])
