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
                      turn-in-place from 45 deg, reading the mission's pseudo-fix.

Not here, by hand -- it is the only writer of /cmd_vel, so killing it takes the rover back:

    ros2 run erc_inference image_checkpoint_controller_node
    ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

The rover must stand next to the orange Start cone, facing east along the south corridor
(the track file's start pose) when the goal is sent.

mission:=track -- the older indoor_mission_node (checkpoint coordinates from the track file,
dead reckoning only); run `ros2 run erc_inference indoor_mission_node` instead.

model:=mppi -- Nav2's MPPI drives instead of the carrot (controller_server + its local costmap
from the profile, and nav2_route_follower_node, as in mission_nav2.launch.py), with this unit's
measured dynamics (config/rover_dynamics.yaml) in the follower's predictor. The mission node must
then run with -p nav2_frames:=true (plants_controller.launch.py does): it publishes the route, the
TF leg_local -> rover_leg and the profile in rover_leg that MPPI works in. In simulation only so
far (rover_simulation/e2e/run_e2e_plants.sh with CONTROLLER=mppi).

Arguments: mission:=images|track, model:=none|mppi, perception:=false|true
(forced on with mission:=images), depth_backend:=unidepth|da3, cone_height_m:=0.23.
target:=cones|plants: with plants, object_detector_node (YOLO26) runs instead of the cone
detector and the mission node is started as the search mission (see `plants` below).

LIVE operation only; it starts the SDK bridge.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            LogInfo)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EqualsSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from erc_inference.nav2_indoor import indoor_params_file

# Must match indoor_mission_node's fix_topic / heading_topic defaults.
INDOOR_FIX_TOPIC = '/erc/indoor/fix'
INDOOR_HEADING_TOPIC = '/erc/indoor/heading_deg'


def generate_launch_description():

    controller_yaml = os.path.join(get_package_share_directory('erc_inference'), 'config', 'controller.yaml')
    args = [
        DeclareLaunchArgument('mission', default_value='images',
                              description="'images': image_checkpoint_controller_node (the NYU mission: cone "
                                          "images, known corridors); 'track': indoor_mission_node (track file)"),
        DeclareLaunchArgument('model', default_value='none',
                              description="'none': carrot_controller_node; 'mppi': Nav2 MPPI on the "
                                          "mission's route"),
        DeclareLaunchArgument('perception', default_value='false',
                              description='run erc_perception free_space_node; forced on with mission:=images'),
        DeclareLaunchArgument('depth_backend', default_value='unidepth',
                              description="free_space_node depth model: 'unidepth' (UniDepthV2) or 'da3'"),
        DeclareLaunchArgument('cone_height_m', default_value='0.23',
                              description="the cones' real height (m); the detector's range comes from it "
                                          "(9-inch sports cone assumed; nobody could measure the organisers')"),
        DeclareLaunchArgument('yolo_conf', default_value='0.35',
                              description='target:=plants: minimum confidence of a YOLO26 detection'),
        DeclareLaunchArgument('turn_enter_deg', default_value='45.0',
                              description='the carrot turns in place once the route is this far off the heading'),
        DeclareLaunchArgument('turn_exit_deg', default_value='15.0',
                              description='... and drives on again once it is within this'),
        DeclareLaunchArgument('target', default_value='cones',
                              description="what mission:=images looks for: 'cones' (cone_detector_node -> /erc/cones) "
                                          "or 'plants' (object_detector_node, YOLO26 -> /erc/objects)"),
    ]

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

    inference = [
        # No model: the carrot drives, as in mission_carrot. Indoors at 0.25 m/s (the floor the
        # rover moves at), and turning in place once the carrot is 45 deg off rather than 90:
        # every corner here is 90 deg in a 2 m corridor (indoor_sim.py). Obstacles are the
        # mission node's (safety envelope + local replanning); the carrot's own side-step stays
        # off -- in 2 m corridors it braked on walls and gave up (cone_sim.py).
        Node(package='erc_inference', executable='carrot_controller_node', name='carrot_controller_node',
             output='screen',
             parameters=[controller_yaml, {'gps_topic': INDOOR_FIX_TOPIC, 'compass_topic': INDOOR_HEADING_TOPIC,
                                           'max_linear_vel': 0.25,
                                           'goal_turn.enter_deg': ParameterValue(LaunchConfiguration('turn_enter_deg'),
                                                                                 value_type=float),
                                           'goal_turn.exit_deg': ParameterValue(LaunchConfiguration('turn_exit_deg'),
                                                                                value_type=float)}],
             condition=IfCondition(EqualsSubstitution(LaunchConfiguration('model'), 'none'))),
    ]

    # Nav2's MPPI instead of the carrot. The follower's predictor and command snapping take the
    # unit's measured dynamics: the delay as one lag (dead time + the mean of the two time
    # constants: what a steady command takes to show), the speed gain, and the turn gains at the
    # speed it will be asked (moving: k_w0 + k_w1 * 0.25; in place: the dead zone and the slope,
    # which no single gain stands for). k_w_auto then follows the moving gain, within [0.2, 2.0].
    share = get_package_share_directory('erc_inference')
    nav2_yaml = indoor_params_file(os.path.join(share, 'config', 'nav2_mppi.yaml'))     # see nav2_indoor.py
    with open(os.path.join(share, 'config', 'rover_dynamics.yaml')) as fh:
        dyn = yaml.safe_load(fh)
    is_mppi = IfCondition(EqualsSubstitution(LaunchConfiguration('model'), 'mppi'))
    mppi = [
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             output='screen', parameters=[nav2_yaml],
             remappings=[('cmd_vel', '/nav2/cmd_vel')], condition=is_mppi),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_mppi', output='screen', parameters=[nav2_yaml], condition=is_mppi),
        Node(package='erc_inference', executable='nav2_route_follower_node', name='nav2_route_follower_node',
             output='screen', condition=is_mppi,
             parameters=[{
                 'predict_delay_s': round(dyn['dead_time_s'] + 0.5 * (dyn['tau_v_s'] + dyn['tau_w_s']), 2),
                 'k_v': float(dyn['k_v']),
                 'k_w_moving': round(dyn['k_w0'] + dyn['k_w1'] * 0.25, 2),
                 'w_rest_deadband': float(dyn['w_rest_deadband']),
                 'w_rest_slope': float(dyn['w_rest_slope']),
                 'v_breakaway': max(0.05, float(dyn['v_floor'])),
                 # this unit moves from 0.10 m/s, and plants_controller sets the envelope's floor there:
                 # at 0.25 MPPI's 0.07 m/s became 0.25 at the same curvature, and the turn saturated
                 # (14-oct). The turn in
                 # place is the follower's own 50 / 20 deg, on the PREDICTED heading: with the carrot's
                 # 30 / 10 it swung +-30 deg about the route and turned in place 43 % of the time (e2e)
                 'v_floor': 0.10,
                 # 0.2-2.0 let the learned gain fall to 0.20 on 14-oct, commands x5, the turn saturated
                 'k_w_bounds': [0.7, 1.6]}]),
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
                 output='screen',
                 condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mission'), "' == 'images' and '",
                                                         LaunchConfiguration('target'), "' == 'cones'"])),
                 parameters=[{'cone_height_m': ParameterValue(LaunchConfiguration('cone_height_m'), value_type=float)}])

    # Potted plants instead (GPU, YOLO26 in the main venv; its boxes on /erc/perception/objects/compressed).
    # The mission node then needs: -p cones_topic:='' -p objects_topic:=/erc/objects
    # -p goals_file:=<share>/config/indoor_plants_goals.yaml -p patrol.max_shows:=0, and either
    # -p explore:=true (no known map: it builds one from /erc/free_space, explore_search.py) or
    # -p route_file:=<your corridors> (the patrol on a known loop).
    # 'person' too: the mission keeps people off its map and waits for one in the way (16-oct)
    plants = ExecuteProcess(cmd=['ros2', 'run', 'erc_perception', 'yolo_wrapper', '--ros-args',
                                 '-p', ['conf:=', LaunchConfiguration('yolo_conf')],
                                 '-p', "classes:=['potted plant', 'person']"], output='screen',
                            condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mission'), "' == 'images' and '",
                                                                    LaunchConfiguration('target'), "' == 'plants'"])))

    reminder = LogInfo(msg=(
        'mission_indoor up (mission=', LaunchConfiguration('mission'), ', model=', LaunchConfiguration('model'),
        '). Then: ros2 run erc_inference image_checkpoint_controller_node  (mission:=track: indoor_mission_node)'
        '  and  ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission '
        '"{resume_from_latest_scanned: false}"'))

    return LaunchDescription(args + bridge + inference + mppi + [localization, perception, cones, plants, reminder])
