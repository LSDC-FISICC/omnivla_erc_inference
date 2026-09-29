"""Off-road image-goal missions, no GPS: drive to where the goal photo was taken.

ERC 2026 off-road track: trails, slopes, gravel, rocks (the Verti-Arena kind of arena);
each mission is a target image, known only on the day. Plan and evidence:
Earth-rover-ros2-bridge/docs/PLAN_OFFROAD.md.

    erc_bridge        the four bridge nodes (camera, telemetry, control, status)
    erc_localization  localization_LOCAL.launch.py: wheels + gyro -> /erc/odometry/local.
                      Not _global: it blocks on the first GPS fix and kills the launch.
    erc_perception    free_space_node (UniDepthV2 by default) -> /erc/free_space, with
                      drop_detection ON by default: it stops at the top of every descent,
                      which here is the safe side of a 0.7 m drop.
    erc_inference     carrot_controller_node (model:=none, default) at 0.25 m/s with
                      turn-in-place from 45 deg, reading the mission's pseudo-fix; or an
                      OmniVLA node (model:=edge|original) remapped the same way.

    erc_perception    flag_detector_node -> /erc/flags: the blue checkpoint flags (CPU).

Not here, by hand -- it is the only writer of /cmd_vel, so killing it takes the rover back.
The organisers (29-sept): three blue flags are the checkpoints, within 1 m of each:

    ros2 run erc_inference flag_checkpoint_node --ros-args -p goal_image:=cp1.jpg,cp2.jpg,cp3.jpg

(the photos are optional hints of which flag is which; without them -p checkpoints:=3). The
earlier photo-homing mission, for a goal that is a view and not a flag:

    ros2 run erc_inference image_goal_offroad_node --ros-args -p goal_image:=/path/goal.jpg
    ros2 action send_goal --feedback /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

goal_image here only goes into the reminder printed at the end, so the command can be copied.

Arguments: model:=none|edge|original, depth_backend:=unidepth|da3, drop_detection:=true|false,
clearance_m:= (empty: perception.yaml's 0.045 m, the rover's ground clearance),
flag_height_m:=0.15 (the flag cloth's vertical extent, assumed; the mission refits it),
goal_image:=/path.jpg (reminder only).

LIVE operation only; it starts the SDK bridge.
"""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Must match indoor_mission_node's fix_topic / heading_topic defaults (the off-road node
# publishes the same pseudo-fix).
FIX_TOPIC = '/erc/indoor/fix'
HEADING_TOPIC = '/erc/indoor/heading_deg'


def generate_launch_description():
    lib = os.path.join(get_package_prefix('erc_inference'), 'lib', 'erc_inference')
    controller_yaml = os.path.join(get_package_share_directory('erc_inference'), 'config', 'controller.yaml')
    args = [
        DeclareLaunchArgument('model', default_value='none',
                              description="'none': carrot_controller_node (no model); 'edge' (OmniVLA-edge) or "
                                          "'original' (OmniVLA 7B)"),
        DeclareLaunchArgument('depth_backend', default_value='unidepth',
                              description="free_space_node depth model: 'unidepth' (UniDepthV2) or 'da3'"),
        DeclareLaunchArgument('drop_detection', default_value='true',
                              description="report drop-offs as obstacles (unidepth only). It also stops at the "
                                          "top of descents: false only where the arena has no drops"),
        DeclareLaunchArgument('clearance_m', default_value='',
                              description="lowest height (m) that counts as an obstacle; empty = perception.yaml "
                                          "(0.045, the rover's ground clearance)"),
        DeclareLaunchArgument('flag_height_m', default_value='0.15',
                              description="the flag cloth's vertical extent (m): ASSUMED; ranges scale with it "
                                          "until the mission fits it from odometry"),
        DeclareLaunchArgument('goal_image', default_value='/path/to/goal.jpg',
                              description='only for the reminder printed at the end: the mission node takes it'),
    ]

    bridge = [
        Node(package='erc_bridge', executable='erc_data_node', name='erc_data_node', output='screen'),
        Node(package='erc_bridge', executable='erc_screenshot_node', name='erc_screenshot_node', output='screen'),
        Node(package='erc_bridge', executable='erc_control_node', name='erc_control_node', output='screen'),
        Node(package='erc_bridge', executable='erc_status_node', name='erc_status_node', output='screen'),
    ]

    remap = ['--ros-args', '-p', f'gps_topic:={FIX_TOPIC}', '-p', f'compass_topic:={HEADING_TOPIC}']
    inference = [
        # As indoors: 0.25 m/s (the floor), turn in place from 45 deg; obstacles are the mission
        # node's (local planner + tilt), the carrot's own side-step stays off.
        Node(package='erc_inference', executable='carrot_controller_node', name='carrot_controller_node',
             output='screen',
             parameters=[controller_yaml, {'gps_topic': FIX_TOPIC, 'compass_topic': HEADING_TOPIC,
                                           'max_linear_vel': 0.25, 'goal_turn.enter_deg': 45.0,
                                           'goal_turn.exit_deg': 15.0}],
             condition=IfCondition(EqualsSubstitution(LaunchConfiguration('model'), 'none'))),
        ExecuteProcess(cmd=[os.path.join(lib, 'omni_vla_wrapper')] + remap, output='screen',
                       condition=IfCondition(EqualsSubstitution(LaunchConfiguration('model'), 'edge'))),
        ExecuteProcess(cmd=[os.path.join(lib, 'omni_vla_original_wrapper')] + remap, output='screen',
                       condition=IfCondition(EqualsSubstitution(LaunchConfiguration('model'), 'original'))),
    ]

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('erc_localization'), 'launch', 'localization_local.launch.py')))

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('erc_perception'), 'launch', 'free_space.launch.py')),
        launch_arguments={'depth_backend': LaunchConfiguration('depth_backend'),
                          'drop_detection': LaunchConfiguration('drop_detection'),
                          'clearance_m': LaunchConfiguration('clearance_m')}.items())

    flags = Node(package='erc_perception', executable='flag_detector_node', name='flag_detector_node',
                 output='screen',
                 parameters=[{'flag_height_m': ParameterValue(LaunchConfiguration('flag_height_m'), value_type=float)}])

    reminder = LogInfo(msg=(
        'mission_offroad up (model=', LaunchConfiguration('model'), ', drop_detection=',
        LaunchConfiguration('drop_detection'), '). Then: ros2 run erc_inference flag_checkpoint_node '
        '--ros-args -p goal_image:=', LaunchConfiguration('goal_image'),
        '  and  ros2 action send_goal --feedback /start_mission erc_inference_msgs/action/StartMission '
        '"{resume_from_latest_scanned: false}"'))

    return LaunchDescription(args + bridge + inference + [localization, perception, flags, reminder])
