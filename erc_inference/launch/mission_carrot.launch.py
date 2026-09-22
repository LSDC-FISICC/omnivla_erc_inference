"""mission.launch.py without OmniVLA: carrot-following polar control, nothing learned in the loop.

Why. The replay of mission_wuhan_hard ran every recorded tick through the
controller under steering_source=plan and =carrot and got the same oscillation
to the decimal. The model's applied deviation had median 0.0 deg -- the 8 deg
deadzone swallowed it. It was not steering. Taking it out also frees the GPU
for the free-space profile (85 ms alone vs 205 ms sharing the device).

Layers:

    erc_bridge        erc_data_node, erc_screenshot_node, erc_control_node,
                      erc_status_node (read-only latency instrumentation)
    erc_static_map    erc_static_map_node, erc_astar_planner_node
    erc_inference     carrot_controller_node   <- instead of omni_vla_wrapper
    erc_perception    free_space_node, SHADOW  (arg perception, default true)
    erc_localization  localization_global.launch.py

Arguments:

    perception:=true|false     run erc_perception's free-space profile. Shadow:
                               it only publishes /erc/free_space for the bag.
    obstacle_stop:=false|true  let carrot_controller_node BRAKE when that
                               profile sees something close straight ahead.
                               Brakes, never steers -- see its docstring. Needs
                               perception:=true. Never run in the field yet.
    obstacle_sidestep:=false|true  instead of stopping for good: brake, turn 90
                               deg in place to the open side, advance 1.3 m,
                               back to the carrot. Replaces obstacle_stop.
                               Needs perception:=true. Never run in the field.

Still by hand, exactly as with mission.launch.py -- and this is where the carrot
distance goes, because it is checkpoint_controller_node's parameter:

    ros2 run erc_inference checkpoint_controller_node --ros-args -p carrot_distance_m:=2.5
    ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

2.5 m, not the 1.5 m default: 1.5 was chosen to keep the MODEL's goal inside its
training distribution (median 1.6 m). Without the model that constraint is gone,
and in simulation 2.5 m halves the weave (3.0 -> 1.6 sign changes/min) at the
cost of p95 cross-track 0.98 -> 1.17 m from cutting corners. Do not use 2.5 m
with mission.launch.py: it would push the model's goal out of range.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Same reason as mission.launch.py: localization_global blocks on the first GPS
# fix, which only arrives once the bridge is up.
LOCALIZATION_DELAY_S = 5.0


def generate_launch_description():
    controller_yaml = os.path.join(
        get_package_share_directory('erc_inference'), 'config', 'controller.yaml')

    args = [
        DeclareLaunchArgument('perception', default_value='true',
                              description='run erc_perception free_space_node (shadow)'),
        DeclareLaunchArgument('obstacle_stop', default_value='false',
                              description='brake on /erc/free_space; never field-tested'),
        DeclareLaunchArgument('stop_distance_m', default_value='1.2',
                              description='must cover the 1.3 s loop delay; 0.8 m hit in simulation'),
        DeclareLaunchArgument('obstacle_sidestep', default_value='false',
                              description='side-step around obstacles (sidestep.py); never field-tested'),
    ]

    bridge = [
        Node(package='erc_bridge', executable='erc_data_node',
             name='erc_data_node', output='screen'),
        Node(package='erc_bridge', executable='erc_screenshot_node',
             name='erc_screenshot_node', output='screen'),
        Node(package='erc_bridge', executable='erc_control_node',
             name='erc_control_node', output='screen'),
        Node(package='erc_bridge', executable='erc_status_node',
             name='erc_status_node', output='screen'),
    ]

    static_map = [
        Node(package='erc_static_map', executable='erc_static_map_node',
             name='erc_static_map_node', output='screen'),
        Node(package='erc_static_map', executable='erc_astar_planner_node',
             name='erc_astar_planner_node', output='screen'),
    ]

    # A plain Node, no wrapper: there is no model, so no conda/venv to enter.
    # controller.yaml's /** block supplies the control parameters exactly as it
    # does for the model nodes; the node forces steering_source to carrot.
    controller = Node(
        package='erc_inference', executable='carrot_controller_node',
        name='carrot_controller_node', output='screen',
        parameters=[controller_yaml, {
            # bool/float through ParameterValue: a bare LaunchConfiguration is a
            # string, and the node declared these as bool/double
            'obstacle_stop': ParameterValue(LaunchConfiguration('obstacle_stop'), value_type=bool),
            'stop_distance_m': ParameterValue(LaunchConfiguration('stop_distance_m'), value_type=float),
            'obstacle_sidestep': ParameterValue(LaunchConfiguration('obstacle_sidestep'), value_type=bool),
        }])

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('erc_perception'), 'launch',
                         'free_space.launch.py')),
        condition=IfCondition(LaunchConfiguration('perception')))

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('erc_localization'), 'launch',
                         'localization_global.launch.py')))

    reminder = LogInfo(msg=(
        'mission_carrot up. Start by hand: ros2 run erc_inference '
        'checkpoint_controller_node --ros-args -p carrot_distance_m:=2.5'))

    return LaunchDescription(
        args + bridge + static_map + [
            controller,
            perception,
            TimerAction(period=LOCALIZATION_DELAY_S, actions=[localization]),
            reminder,
        ])
