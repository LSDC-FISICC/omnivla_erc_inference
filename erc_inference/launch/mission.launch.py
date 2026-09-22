"""Bring up the full mission stack, everything except the controller itself.

Layers, in the order they come up:

    erc_bridge        erc_data_node, erc_screenshot_node, erc_control_node,
                      erc_status_node (read-only latency instrumentation)
    erc_static_map    erc_static_map_node, erc_astar_planner_node
    erc_inference     omnivla edge node (model + motion controller)
    erc_localization  localization_global.launch.py

Two things are deliberately NOT here, and have to be done by hand once this
launch is up:

    ros2 run erc_inference checkpoint_controller_node
    ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \
        "{resume_from_latest_scanned: false}"

The controller stays out because it is the only node that writes /cmd_vel, so
killing it is how an operator takes the rover back for an intervention. If it
were launched here, Ctrl-C would take the whole stack down with it and the
mission would have to be restarted from the bridge up.

This launch is for LIVE field operation only. It starts the SDK bridge, which
has nothing to do on a bag, so there is no use_sim_time argument: for replay,
run localization_global.launch.py with use_sim_time:=true on its own.
"""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# localization_global.launch.py resolves the magnetic declination inside an
# OpaqueFunction that BLOCKS on the first live /erc/gps fix. A launch file
# visits its top-level actions synchronously, so that wait blocks the asyncio
# loop that spawns processes -- including the bridge that publishes the very
# fix it is waiting for. Declaring the bridge first is not enough; verified
# empirically, the bridge process is not spawned until the blocking function
# returns, so the wait always hits its 30 s timeout and kills the launch.
# Deferring the include behind a timer lets the bridge spawn and start polling
# first. Raise this if the bridge needs longer to reach the SDK.
LOCALIZATION_DELAY_S = 5.0


def generate_launch_description():
    # Loads config/controller.yaml and fails loudly if it is missing, which
    # omni_vla_edge_wrapper does not do. Run through a wrapper rather than as a
    # Node because the model needs its conda environment, which a plain exec of
    # the entry point does not get.
    omnivla_wrapper = os.path.join(
        get_package_prefix('erc_inference'), 'lib', 'erc_inference', 'omni_vla_wrapper')

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

    # Brings up the local layer itself (static transforms, attitude filter,
    # local EKF) on top of navsat_transform and the global EKF -- see its own
    # docstring. Including localization_local here too would start those twice.
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('erc_localization'), 'launch',
                         'localization_global.launch.py')))

    return LaunchDescription(
        bridge + static_map + [
            inference,
            TimerAction(period=LOCALIZATION_DELAY_S, actions=[localization]),
        ])
