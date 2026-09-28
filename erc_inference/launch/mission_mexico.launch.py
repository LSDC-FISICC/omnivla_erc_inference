"""Strategy 5 (carrot + UniDepthV2) set up for grass fields: the Mexico mission.

The same stack as mission_carrot_unidepth.launch.py -- carrot and polar control on the
A* route, the side-step reflex and, with local_replan in terminal 3, the local map --
with the choices the grass evidence points to (docs COMANDOS_MISION, 25-28 sept):

  depth_backend unidepth  UniDepthV2 does not compress what is far (DA3 put a tree line
                          20 m away at 0.4-1.1 m in Panama) and fits the ground plane per
                          frame, so it needs no camera pitch -- the measured pitch of the
                          units disagrees with the config (TAREA1 7.6, open).
  obstacle_mode height    'contact' depends on that pitch; not used here.
  obstacle_sidestep true  brake and side-step on what perception reports.
  clearance_m             the lowest height that counts as an obstacle. Default 0.045 (the
                          rover's ground clearance), the only value validated on real
                          obstacles. Its risk in a grass field is the opposite of getting
                          stuck: medium grass (5-10 cm) above 4.5 cm gets flagged everywhere
                          and the rover brakes and side-steps until it gives up. If it does
                          that on grass it could drive through, relaunch with e.g.
                          clearance_m:=0.08 -- nothing lower than that is then seen.
                          The node reads it at start; changing it needs a relaunch.

    ros2 launch erc_inference mission_mexico.launch.py
    ros2 launch erc_inference mission_mexico.launch.py clearance_m:=0.08

Every mission_carrot argument can still be given (stop_distance_m, obstacle_stop, ...).
Still by hand, as in every mission launch -- terminal 3 is what the operator kills:

    ros2 run erc_inference checkpoint_controller_node --ros-args \\
        -p carrot_distance_m:=1.5 -p local_replan:=true
    ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \\
        "{resume_from_latest_scanned: false}"

Before it: magnetometer calibration on that unit at the site, GPS fix, and
`ros2 run erc_static_map prefetch_osm` with the SDK up. Record with `ros2 bag record -a`
and note the time and the kind of grass (short / medium / tall, passed or stuck) of every
odd stop: no bag has tall grass seen from a distance yet. There is no automatic
"wheels turn but the rover does not move" detector: someone has to watch for it getting
stuck in grass and Ctrl-C terminal 3.

Needs ~/lsdc/erc-omni-vla/.venv-unidepth (UniDepthV2, weights cached: no internet needed).
UniDepthV2 takes ~6 s to load after the first image: wait until /omnivla_debug stops
saying "perception stale" and `ros2 topic hz /erc/free_space` shows ~3 Hz.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_directory
    return LaunchDescription([
        DeclareLaunchArgument('clearance_m', default_value='0.045',
                              description='lowest height (m) that counts as an obstacle; 0.045 = '
                                          'ground clearance (validated). Raise (e.g. 0.08) only if '
                                          'the rover brakes on grass it could drive through'),
        DeclareLaunchArgument('obstacle_sidestep', default_value='true',
                              description='brake and side-step on what perception reports'),
        LogInfo(msg=['mission_mexico: carrot + UniDepthV2 (height), clearance_m=',
                     LaunchConfiguration('clearance_m')]),
        # the carrot stack without its own perception: this launch brings it up below with
        # the grass parameter, which mission_carrot does not forward
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                share('erc_inference'), 'launch', 'mission_carrot.launch.py')),
            launch_arguments={'perception': 'false',
                              'obstacle_sidestep': LaunchConfiguration('obstacle_sidestep')}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                share('erc_perception'), 'launch', 'free_space.launch.py')),
            launch_arguments={'depth_backend': 'unidepth',
                              'obstacle_mode': 'height',
                              # off on purpose: on rolling grass it would stop at every hill
                              # top and descent (the camera cannot tell them from a drop)
                              'drop_detection': 'false',
                              'clearance_m': LaunchConfiguration('clearance_m')}.items()),
    ])
