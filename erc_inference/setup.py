from setuptools import find_packages, setup

package_name = 'erc_inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/mission_carrot.launch.py', 'launch/mission_nav2.launch.py', 'launch/mission_carrot_unidepth.launch.py', 'launch/mission_nav2_unidepth.launch.py', 'launch/mission_mexico.launch.py', 'launch/mission_indoor.launch.py', 'launch/mission_offroad.launch.py', 'launch/mission_plants.launch.py', 'launch/plants_controller.launch.py']),
        ('share/' + package_name + '/config', ['config/controller.yaml', 'config/nav2_mppi.yaml', 'config/indoor_nyu_track.yaml',
                                               'config/indoor_nyu_goals.yaml', 'config/indoor_plants_goals.yaml', 'config/rover_dynamics.yaml',
                                               'config/offroad_arena.yaml']),
        ('share/' + package_name + '/config/goal_images/nyu', ['config/goal_images/nyu/cp1.jpg', 'config/goal_images/nyu/cp2.jpg',
                                                               'config/goal_images/nyu/cp3.jpg', 'config/goal_images/nyu/cp4.jpg',
                                                               'config/goal_images/nyu/start_finish.jpg']),
        ('share/' + package_name + '/rviz', ['rviz/local_planning.rviz']),
        ('share/' + package_name + '/foxglove', ['foxglove/planning_tabs.json', 'foxglove/robot_live.json', 'foxglove/plants_search.json', 'foxglove/excitation.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jabes',
    maintainer_email='jabes@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'checkpoint_controller_node = erc_inference.checkpoint_controller_node:main',
            'carrot_controller_node = erc_inference.carrot_controller_node:main',
            'nav2_route_follower_node = erc_inference.nav2_route_follower_node:main',
            'indoor_mission_node = erc_inference.indoor_mission_node:main',
            'image_checkpoint_controller_node = erc_inference.image_checkpoint_controller_node:main',
            'local_map_node = erc_inference.local_map_node:main',
            'image_goal_offroad_node = erc_inference.image_goal_offroad_node:main',
            'flag_checkpoint_node = erc_inference.flag_checkpoint_node:main',
        ],
    },
)
