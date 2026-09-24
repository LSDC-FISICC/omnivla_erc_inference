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
        ('share/' + package_name + '/launch', ['launch/omnivla_edge.launch.py', 'launch/omnivla_original.launch.py', 'launch/mission.launch.py', 'launch/mission_omnivla.launch.py', 'launch/mission_carrot.launch.py', 'launch/mission_nav2.launch.py']),
        ('share/' + package_name + '/config', ['config/controller.yaml', 'config/nav2_mppi.yaml']),
        ('share/' + package_name + '/rviz', ['rviz/local_planning.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    scripts=['scripts/omni_vla_wrapper',
             'scripts/omni_vla_edge_wrapper',
             'scripts/omni_vla_original_wrapper'], # Instala el ejecutable envuelto
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
            'omni_vla_edge_node = erc_inference.omni_vla_edge_node:main',
            'omnivla_edge_node = erc_inference.omnivla_edge_node:main',
            'omnivla_original_node = erc_inference.omnivla_original_node:main',
            'checkpoint_controller_node = erc_inference.checkpoint_controller_node:main',
            'carrot_controller_node = erc_inference.carrot_controller_node:main',
            'nav2_route_follower_node = erc_inference.nav2_route_follower_node:main',
        ],
    },
)
