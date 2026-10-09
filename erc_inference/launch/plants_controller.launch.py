"""The node that drives the potted-plant search: image_checkpoint_controller_node as the
exploring search mission (erc_inference.explore_search). Terminal 3, after
mission_plants.launch.py; it is the only writer of /cmd_vel, so Ctrl-C here stops the rover.

    ros2 launch erc_inference plants_controller.launch.py                  # 3 plants, 30 minutes
    ros2 launch erc_inference plants_controller.launch.py plants:=5 timeout_s:=900
    ros2 action send_goal --feedback /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

No known map: the rover's pose when the goal is sent is the map's origin (0, 0, facing +x).
The map and the plants on it are published on /erc/explore/map and /erc/explore/objects.

Arguments:
  plants       how many potted plants to show before the mission ends (default 3)
  timeout_s    the mission's time limit (default 1800)
  show_s       how long it holds still in front of each plant (default 5)
  arrive_m     how close it gets before showing it (default 1.8)
  stop_when_explored  true (default): also end when nothing is left to explore
More of the node's parameters: `ros2 run erc_inference image_checkpoint_controller_node
--ros-args -p ...` with the ones set below.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('erc_inference')

    def arg(name, kind):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)

    controller = Node(
        package='erc_inference', executable='image_checkpoint_controller_node',
        name='image_checkpoint_controller_node', output='screen', emulate_tty=True,
        parameters=[{
            'explore': True,                       # build the map; no known corridors
            'cones_topic': '',                     # only objects: a cone false positive would be visited too
            'objects_topic': '/erc/objects',
            'object_classes': ['potted plant'],
            'goals_file': os.path.join(share, 'config', 'indoor_plants_goals.yaml'),
            'patrol.max_shows': arg('plants', int),
            'patrol.show_s': arg('show_s', float),
            'patrol.arrive_m': arg('arrive_m', float),
            'explore.stop_when_explored': arg('stop_when_explored', bool),
            'mission_timeout_s': arg('timeout_s', float),
            # its own file: a state saved by the cone mission must never be resumed here
            'state_file': os.path.expanduser('~/.ros/erc_plants_mission_state.json'),
        }])
    return LaunchDescription([
        DeclareLaunchArgument('plants', default_value='3', description='potted plants to show before it ends'),
        DeclareLaunchArgument('timeout_s', default_value='1800.0', description='mission time limit (s)'),
        DeclareLaunchArgument('show_s', default_value='5.0', description='hold still this long at each plant (s)'),
        DeclareLaunchArgument('arrive_m', default_value='1.8', description='show a plant from this distance (m)'),
        DeclareLaunchArgument('stop_when_explored', default_value='true',
                              description='end the mission when no frontier is left'),
        controller,
    ])
