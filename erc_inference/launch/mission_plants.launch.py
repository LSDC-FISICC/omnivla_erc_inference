"""Indoor search mission: explore with no known map and find the potted plants. Terminal 2.

Everything the mission needs except the node that drives -- mission_indoor.launch.py with
target:=plants:

    erc_bridge        camera, telemetry, control, status (the SDK must be running)
    erc_localization  localization_local: wheels + gyro -> /erc/odometry/local (no GPS)
    erc_perception    free_space_node (UniDepthV2) -> /erc/free_space, the map's input;
                      object_detector_node (YOLO26, 'potted plant') -> /erc/objects
    erc_inference     carrot_controller_node at 0.25 m/s, turning in place from 30 deg -- or, with
                      controller:=mppi, Nav2's MPPI on the mission's route (controller_server,
                      its local costmap from the profile, nav2_route_follower_node)
                      local_map_node: /erc/local_map (+ /objects, /path), the map and the plants
                      on it as the rover would build them, to look at before anything drives

Then, in its own terminal -- it is the only writer of /cmd_vel, so Ctrl-C there stops the rover:

    ros2 launch erc_inference plants_controller.launch.py
    ros2 action send_goal --feedback /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

Arguments: controller:=carrot|mppi (default carrot), record:=true (a rosbag of the run, rosbags/mission_<time>_macetas),
local_map:=true|false (default true), debug_images:=true|false (default true: the frame with the ground and the obstacle
band, and the model's distance, as JPEGs; the YOLO26 boxes are always published),
foxglove:=true|false (default false: also start foxglove_bridge on 127.0.0.1:foxglove_port,
8766 -- forward the port and open ws://localhost:8766 as "Foxglove WebSocket"; layout
share/erc_inference/foxglove/plants_search.json; the raw camera images are not offered, look at
/erc/front_camera/compressed), yolo_conf:=0.35 (the detector's minimum
confidence), depth_backend:=unidepth|da3.

LIVE operation only; it starts the SDK bridge.
"""
import os
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('erc_inference')
    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'mission_indoor.launch.py')),
        launch_arguments={'mission': 'images', 'target': 'plants',
                          'model': PythonExpression(["'mppi' if '", LaunchConfiguration('controller'),
                                                     "' == 'mppi' else 'none'"]),
                          'depth_backend': LaunchConfiguration('depth_backend'),
                          'yolo_conf': LaunchConfiguration('yolo_conf'),
                          # Turning in place from 30 deg, not 45: on this rover turning WHILE driving is the
                          # part that does not repeat (it did 0.9x of the command in one excitation run and
                          # 0.4x in the next, config/rover_dynamics.yaml), turning in place is steady. In the
                          # ROS e2e with its measured dynamics: 45/15 -> 4 and 10 touches, and a door jamb it
                          # never got past when the gain was the low one; 30/10 -> 0 and 5, and through.
                          'turn_enter_deg': LaunchConfiguration('turn_enter_deg'),
                          'turn_exit_deg': LaunchConfiguration('turn_exit_deg'),
                          'debug_images': LaunchConfiguration('debug_images')}.items())
    foxglove = ExecuteProcess(
        cmd=['ros2', 'launch', 'foxglove_bridge', 'foxglove_bridge_launch.xml',
             ['port:=', LaunchConfiguration('foxglove_port')], 'address:=127.0.0.1',
             # every topic but the two raw camera images: 1.77 MB a frame, 7 MB/s, and through an SSH
             # tunnel Foxglove ran ~20 s behind showing one (2026-10-09). Their JPEGs are
             # /erc/{front,rear}_camera/compressed; the YOLO26 boxes, /erc/perception/objects/compressed.
             "topic_whitelist:=['^(?!/erc/(front|rear)_camera$).*']"],
        output='log', condition=IfCondition(LaunchConfiguration('foxglove')))
    # The map the rover would build, to look at before anything drives: view only, it commands nothing.
    local_map = Node(package='erc_inference', executable='local_map_node', name='local_map_node', output='screen',
                     parameters=[{'map_forget_m': ParameterValue(LaunchConfiguration('map_forget_m'), value_type=float)}],
                     condition=IfCondition(LaunchConfiguration('local_map')))
    # Everything but the raw camera images and the distance image (GBs a minute), in rosbags/.
    # Started and closed by this launch, so Ctrl-C here closes the bag properly.
    bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '--all-topics',
             '--exclude-regex', '/erc/(front|rear)_camera$|/erc/perception/distance$',
             '-o', [LaunchConfiguration('bag_dir'), '/mission_', LaunchConfiguration('bag_name')]],
        output='log', condition=IfCondition(LaunchConfiguration('record')))
    reminder = LogInfo(msg=(
        'mission_plants up. Then, in another terminal:  ros2 launch erc_inference plants_controller.launch.py'
        '  and  ros2 action send_goal --feedback /start_mission erc_inference_msgs/action/StartMission '
        '"{resume_from_latest_scanned: false}"'))
    return LaunchDescription([
        DeclareLaunchArgument('controller', default_value='carrot',
                              description="what follows the mission's route: 'carrot' (carrot_controller_node) or "
                                          "'mppi' (Nav2 MPPI; in simulation only so far)"),
        DeclareLaunchArgument('debug_images', default_value='true',
                              description='free_space_node also publishes /erc/perception/{overlay,depth}/compressed'),
        DeclareLaunchArgument('depth_backend', default_value='unidepth',
                              description="free_space_node depth model: 'unidepth' (UniDepthV2) or 'da3'"),
        DeclareLaunchArgument('yolo_conf', default_value='0.5',
                              description='minimum confidence of a YOLO26 detection (raise it if it sees plants that are not)'),
        DeclareLaunchArgument('local_map', default_value='true',
                              description='also run local_map_node: /erc/local_map, the map with nobody driving'),
        DeclareLaunchArgument('map_forget_m', default_value='25.0',
                              description='local_map_node: keep what was seen in the last this-many metres driven (0: all)'),
        DeclareLaunchArgument('turn_enter_deg', default_value='30.0',
                              description='the carrot turns in place once the route is this far off the heading'),
        DeclareLaunchArgument('turn_exit_deg', default_value='10.0',
                              description='... and drives on again once it is within this'),
        DeclareLaunchArgument('record', default_value='false',
                              description='record a rosbag of the whole run (all but the raw images) in bag_dir'),
        DeclareLaunchArgument('bag_dir', default_value=os.path.expanduser('~/lsdc/erc-omni-vla/rosbags')),
        DeclareLaunchArgument('bag_name', default_value=time.strftime('%m%d_%H%M') + '_macetas'),
        DeclareLaunchArgument('foxglove', default_value='false',
                              description='also start foxglove_bridge (127.0.0.1 only)'),
        DeclareLaunchArgument('foxglove_port', default_value='8766'),
        stack, local_map, foxglove, bag, reminder,
    ])
