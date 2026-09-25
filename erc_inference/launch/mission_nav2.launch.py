"""mission_carrot.launch.py with Nav2's MPPI as the local layer, on a costmap from DA3.

Same architecture, one layer swapped:

    global   checkpoint_controller_node: A* on OSM per leg (unchanged)
    local    Nav2 controller_server: MPPI on a local_costmap whose obstacle layer
             is fed by erc_perception's DA3 free-space profile
             (instead of carrot + polar control + side-step)
    output   checkpoint_controller_node gates and ramps /cmd_vel (unchanged)

Layers:

    erc_bridge        erc_data_node, erc_screenshot_node, erc_control_node, erc_status_node
    erc_static_map    erc_static_map_node, erc_astar_planner_node
    erc_perception    free_space_node (DA3) -- required here, not shadow
    erc_localization  localization_global.launch.py
    nav2              controller_server (MPPI + local_costmap), lifecycle_manager
    erc_inference     nav2_route_follower_node: route -> FollowPath, profile ->
                      costmap, MPPI's command -> /omnivla/cmd_vel

Arguments:

    sim:=false|true   true brings up only the nav2 layer + the follower, for
                      rover_simulation/e2e (fake rover, fake SDK)

Still by hand, as in the other mission launches (checkpoint_controller_node is
what an operator kills to take the rover back):

    ros2 run erc_inference checkpoint_controller_node --ros-args -p carrot_distance_m:=1.5
    ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

The carrot keeps being computed but nothing drives on it. Adding
-p local_replan:=true feeds MPPI the locally replanned route instead of the raw
A* one. Needs Nav2 (see config/nav2_mppi.yaml). Never run in the field.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

LOCALIZATION_DELAY_S = 5.0


def generate_launch_description():
    share = get_package_share_directory('erc_inference')
    nav2_yaml = os.path.join(share, 'config', 'nav2_mppi.yaml')
    real = UnlessCondition(LaunchConfiguration('sim'))

    args = [DeclareLaunchArgument('obstacle_mode', default_value='height',
                              description="free_space_node: 'height' or 'contact' (segmentation + "
                                          "ground contact, against far-tree phantoms)"),
            DeclareLaunchArgument('sim', default_value='false',
                                  description='only the nav2 layer + follower (rover_simulation/e2e)')]

    bridge = [
        Node(package='erc_bridge', executable=exe, name=exe, output='screen', condition=real)
        for exe in ('erc_data_node', 'erc_screenshot_node', 'erc_control_node', 'erc_status_node')
    ]
    static_map = [
        Node(package='erc_static_map', executable=exe, name=exe, output='screen', condition=real)
        for exe in ('erc_static_map_node', 'erc_astar_planner_node')
    ]
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('erc_perception'), 'launch', 'free_space.launch.py')),
        launch_arguments={'obstacle_mode': LaunchConfiguration('obstacle_mode')}.items(),
        condition=real)
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('erc_localization'), 'launch',
                         'localization_global.launch.py')),
        condition=real)

    nav2 = [
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             output='screen', parameters=[nav2_yaml],
             remappings=[('cmd_vel', '/nav2/cmd_vel')]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_mppi', output='screen', parameters=[nav2_yaml]),
        Node(package='erc_inference', executable='nav2_route_follower_node',
             name='nav2_route_follower_node', output='screen'),
    ]

    reminder = LogInfo(msg=(
        'mission_nav2 up. Start by hand: ros2 run erc_inference checkpoint_controller_node '
        '--ros-args -p carrot_distance_m:=1.5   (MPPI follows /erc/global_route)'))

    return LaunchDescription(
        args + bridge + static_map + nav2 + [
            perception,
            TimerAction(period=LOCALIZATION_DELAY_S, actions=[localization]),
            reminder,
        ])
