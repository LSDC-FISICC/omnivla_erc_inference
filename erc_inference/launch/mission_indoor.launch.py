"""Indoor missions, no GPS: the NYU image-goal mission (default) or the track-file mission.

mission:=images (default) -- ERC 2026 NYU indoor: reach, in order, the cones the goal images
show (config/indoor_nyu_goals.yaml, the organisers' photos) on the known corridor loop
(config/indoor_nyu_track.yaml); 30 minutes, autonomous. Plan and evidence:
Earth-rover-ros2-bridge/docs/PLAN_INDOOR_NYU.md.

    erc_bridge        the four bridge nodes (camera, telemetry, control, status)
    erc_localization  localization_LOCAL.launch.py: wheels + gyro -> /erc/odometry/local.
                      Not _global: it blocks on the first GPS fix and kills the launch.
    erc_perception    free_space_node (UniDepthV2 by default) -> /erc/free_space, and
                      cone_detector_node -> /erc/cones. Both needed by the mission.
    erc_inference     carrot_controller_node (model:=none, default) at 0.25 m/s with
                      turn-in-place from 45 deg, reading the mission's pseudo-fix; or an
                      OmniVLA node (model:=edge|original) remapped the same way.

Not here, by hand -- it is the only writer of /cmd_vel, so killing it takes the rover back:

    ros2 run erc_inference image_checkpoint_controller_node
    ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

The rover must stand next to the orange Start cone, facing east along the south corridor
(the track file's start pose) when the goal is sent.

mission:=track -- the older indoor_mission_node (checkpoint coordinates from the track file,
dead reckoning only); run `ros2 run erc_inference indoor_mission_node` instead.

Arguments: mission:=images|track, model:=none|edge|original, perception:=false|true
(forced on with mission:=images), depth_backend:=unidepth|da3, cone_height_m:=0.23.

LIVE operation only; it starts the SDK bridge.
"""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            LogInfo)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EqualsSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Must match indoor_mission_node's fix_topic / heading_topic defaults.
INDOOR_FIX_TOPIC = '/erc/indoor/fix'
INDOOR_HEADING_TOPIC = '/erc/indoor/heading_deg'


def generate_launch_description():
    lib = os.path.join(get_package_prefix('erc_inference'), 'lib', 'erc_inference')

    controller_yaml = os.path.join(get_package_share_directory('erc_inference'), 'config', 'controller.yaml')
    args = [
        DeclareLaunchArgument('mission', default_value='images',
                              description="'images': image_checkpoint_controller_node (the NYU mission: cone "
                                          "images, known corridors); 'track': indoor_mission_node (track file)"),
        DeclareLaunchArgument('model', default_value='none',
                              description="'none': carrot_controller_node (no model); 'edge' (OmniVLA-edge) or "
                                          "'original' (OmniVLA 7B)"),
        DeclareLaunchArgument('perception', default_value='false',
                              description='run erc_perception free_space_node; forced on with mission:=images'),
        DeclareLaunchArgument('depth_backend', default_value='unidepth',
                              description="free_space_node depth model: 'unidepth' (UniDepthV2) or 'da3'"),
        DeclareLaunchArgument('cone_height_m', default_value='0.23',
                              description="the cones' real height (m); the detector's range comes from it "
                                          "(9-inch sports cone assumed; nobody could measure the organisers')"),
    ]
    images = IfCondition(EqualsSubstitution(LaunchConfiguration('mission'), 'images'))

    bridge = [
        Node(package='erc_bridge', executable='erc_data_node',
             name='erc_data_node', output='screen'),
        Node(package='erc_bridge', executable='erc_screenshot_node',
             name='erc_screenshot_node', output='screen'),
        Node(package='erc_bridge', executable='erc_control_node',
             name='erc_control_node', output='screen'),
        # Read-only latency instrumentation, as in mission.launch.py.
        Node(package='erc_bridge', executable='erc_status_node',
             name='erc_status_node', output='screen'),
    ]

    # Both wrappers end in `--ros-args --params-file controller.yaml "$@"`; this
    # second --ros-args block only moves the two input topics.
    remap = ['--ros-args',
             '-p', f'gps_topic:={INDOOR_FIX_TOPIC}',
             '-p', f'compass_topic:={INDOOR_HEADING_TOPIC}']
    # Started early: loading the checkpoint is the slowest thing here.
    inference = [
        # No model: the carrot drives, as in mission_carrot. Indoors at 0.25 m/s (the floor the
        # rover moves at), and turning in place once the carrot is 45 deg off rather than 90:
        # every corner here is 90 deg in a 2 m corridor (indoor_sim.py). Obstacles are the
        # mission node's (safety envelope + local replanning); the carrot's own side-step stays
        # off -- in 2 m corridors it braked on walls and gave up (cone_sim.py).
        Node(package='erc_inference', executable='carrot_controller_node', name='carrot_controller_node',
             output='screen',
             parameters=[controller_yaml, {'gps_topic': INDOOR_FIX_TOPIC, 'compass_topic': INDOOR_HEADING_TOPIC,
                                           'max_linear_vel': 0.25, 'goal_turn.enter_deg': 45.0,
                                           'goal_turn.exit_deg': 15.0}],
             condition=IfCondition(EqualsSubstitution(LaunchConfiguration('model'), 'none'))),
        ExecuteProcess(cmd=[os.path.join(lib, 'omni_vla_wrapper')] + remap, output='screen',
                       condition=IfCondition(EqualsSubstitution(LaunchConfiguration('model'), 'edge'))),
        ExecuteProcess(cmd=[os.path.join(lib, 'omni_vla_original_wrapper')] + remap, output='screen',
                       condition=IfCondition(EqualsSubstitution(LaunchConfiguration('model'), 'original'))),
    ]

    # No GPS wait inside, so no TimerAction as in the outdoor launches.
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('erc_localization'), 'launch',
                         'localization_local.launch.py')))

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('erc_perception'), 'launch',
                         'free_space.launch.py')),
        launch_arguments={'depth_backend': LaunchConfiguration('depth_backend')}.items(),
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('perception'), "' == 'true' or '",
                                                LaunchConfiguration('mission'), "' == 'images'"])))

    # The cones the goal images show (CPU, ~10 ms/frame): /erc/cones for the mission node.
    cones = Node(package='erc_perception', executable='cone_detector_node', name='cone_detector_node',
                 output='screen', condition=images,
                 parameters=[{'cone_height_m': ParameterValue(LaunchConfiguration('cone_height_m'), value_type=float)}])

    reminder = LogInfo(msg=(
        'mission_indoor up (mission=', LaunchConfiguration('mission'), ', model=', LaunchConfiguration('model'),
        '). Then: ros2 run erc_inference image_checkpoint_controller_node  (mission:=track: indoor_mission_node)'
        '  and  ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission '
        '"{resume_from_latest_scanned: false}"'))

    return LaunchDescription(args + bridge + inference + [localization, perception, cones, reminder])
