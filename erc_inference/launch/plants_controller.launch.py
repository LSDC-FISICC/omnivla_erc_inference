"""The node that drives the potted-plant search: image_checkpoint_controller_node as the
exploring search mission (erc_inference.explore_search). Terminal 3, after
mission_plants.launch.py; it is the only writer of /cmd_vel, so Ctrl-C here stops the rover.

    ros2 launch erc_inference plants_controller.launch.py                  # 3 plants, 30 minutes
    ros2 launch erc_inference plants_controller.launch.py plants:=5 timeout_s:=900
    ros2 action send_goal --feedback /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

No known map: the rover's pose when the goal is sent is the map's origin (0, 0, facing +x).
The map and the plants on it are published on /erc/explore/map and /erc/explore/objects.

What it does (since 12-oct): drives on wherever the profile shows an open way ahead that leads to
ground not seen yet, and counts each potted plant as it comes within count_m of it. It does not
turn in place to search, does not go to the plants and does not stop at them. With no open way
ahead the map decides (frontiers, A*); with nowhere to go it stands still 10 s, then makes quarter
turns towards the open side, and a full turn with no way on ends the mission. This rover climbs
what its wheels touch (it stood up a wall and rolled over a post on 12-oct), so the safety
envelope here slows it within slow_m of anything ahead, stops it at stop_m, backs it straight off
at too_close_m and holds it still when the profile is stale. approach:=true is the mission as it was before: a look around at the start, then go
to each plant, face it and hold still show_s.

Arguments:
  plants       how many potted plants to count before the mission ends (default 3)
  timeout_s    the mission's time limit (default 1800)
  count_m      a plant counts once the rover is within this of it, driving by (default 2.0)
  apart_m      a plant this close to one already counted is the same plant (default 1.5)
  slow_m, stop_m, too_close_m   the reflexes above (defaults 1.5, 0.6, 0.35 m)
  approach     false (default). true: go to each plant and show it, with these three:
  show_s       how long it holds still in front of each plant (default 5)
  arrive_m     a plant counts once the rover is within this of it (default 1.0)
  standoff_m   the route to a plant ends this far short of it (default 0.9)
  stop_when_explored  true (default): also end when nothing is left to explore
  forget_m     the map keeps what was seen in the last this-many metres driven (default 25; 0: all)
More of the node's parameters: `ros2 run erc_inference image_checkpoint_controller_node
--ros-args -p ...` with the ones set below.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('erc_inference')

    def arg(name, kind):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)

    approach = PythonExpression(["'", LaunchConfiguration('approach'), "' == 'true'"])
    drive_by = PythonExpression(["'", LaunchConfiguration('approach'), "' != 'true'"])

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
            'patrol.standoff_m': arg('standoff_m', float),
            'patrol.same_class_apart_m': arg('apart_m', float),
            # driving past and counting, or (approach) going to each plant: explore_search.py
            'explore.count_on_sight': ParameterValue(drive_by, value_type=bool),
            'explore.forward_first': ParameterValue(drive_by, value_type=bool),
            'explore.look_around': ParameterValue(approach, value_type=bool),
            'patrol.start_scan': ParameterValue(approach, value_type=bool),
            'explore.count_m': arg('count_m', float),
            # A tilt needs the accelerometer to agree: on 12-oct the attitude filter read 15-17 deg
            # three times while turning in place on a flat floor (the accelerometer: 0.8 deg), and
            # each time the mission backed off and marked an obstacle ahead that was not there.
            'safety.tilt_accel_confirm_deg': 6.0,
            # This rover climbs whatever its wheels touch -- a wall up to vertical, a post onto its
            # side (12-oct, twice). So here contact is not "a slow bump": within 1.5 m of anything
            # ahead it goes at 0.15 m/s, nothing within 0.6 m ahead is driven at (the command takes
            # ~1.3 s to act and the profile is ~0.4 s old: 0.25 m at 0.15 m/s), nothing is turned
            # beside within 0.35 m, and with no profile for 1.5 s nothing moves at all.
            'safety.obstacle_enabled': True,
            'safety.stop_distance_m': arg('stop_m', float),
            'safety.slow_distance_m': arg('slow_m', float),
            'safety.min_moving_linear_vel': 0.10,          # this unit moves from 0.10 m/s (rover_dynamics.yaml)
            'safety.stop_confirm_ticks': 1,
            'safety.stale_stops_turning': True,
            'safety.too_close_m': arg('too_close_m', float),
            # what MPPI needs if it is the one driving (mission_plants.launch.py controller:=mppi);
            # with the carrot nobody listens
            'nav2_frames': True,
            'explore.stop_when_explored': arg('stop_when_explored', bool),
            'explore.map_forget_m': arg('forget_m', float),
            'mission_timeout_s': arg('timeout_s', float),
            # its own file: a state saved by the cone mission must never be resumed here
            'state_file': os.path.expanduser('~/.ros/erc_plants_mission_state.json'),
        }])
    return LaunchDescription([
        DeclareLaunchArgument('plants', default_value='3', description='potted plants to count before it ends'),
        DeclareLaunchArgument('timeout_s', default_value='1800.0', description='mission time limit (s)'),
        DeclareLaunchArgument('approach', default_value='false',
                              description='false: drive past and count; true: go to each plant, face it, hold still'),
        DeclareLaunchArgument('count_m', default_value='2.0',
                              description='driving past: a plant counts once the rover is within this of it (m)'),
        DeclareLaunchArgument('stop_m', default_value='0.6',
                              description='nothing nearer than this within +-20 deg ahead is driven at (m)'),
        DeclareLaunchArgument('slow_m', default_value='1.5',
                              description='0.15 m/s instead of 0.25 with anything nearer than this ahead (m; 0: off)'),
        DeclareLaunchArgument('too_close_m', default_value='0.35',
                              description='with anything nearer than this within +-40 deg: straight back, no turning (0: off)'),
        DeclareLaunchArgument('show_s', default_value='5.0', description='hold still this long at each plant (s)'),
        DeclareLaunchArgument('arrive_m', default_value='1.0',
                              description='a plant counts once the rover is within this of it (m)'),
        DeclareLaunchArgument('standoff_m', default_value='0.9',
                              description='the route to a plant ends this far short of it (m); below arrive_m'),
        DeclareLaunchArgument('apart_m', default_value='1.5',
                              description='a plant this close to one already shown is that same plant: not shown again'),
        DeclareLaunchArgument('forget_m', default_value='25.0',
                              description='the map keeps what was seen in the last this-many metres driven (0: all)'),
        DeclareLaunchArgument('stop_when_explored', default_value='true',
                              description='end the mission when no frontier is left'),
        controller,
    ])
