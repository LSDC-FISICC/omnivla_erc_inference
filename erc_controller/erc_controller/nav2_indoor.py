"""config/nav2_mppi.yaml as the indoor missions use it (mission_indoor.launch.py model:=mppi,
rover_simulation/e2e/run_e2e_plants.sh): the same file with the few values a 2 m corridor needs
changed, written to a temporary file for controller_server and its lifecycle manager.

    python3 -m erc_controller.nav2_indoor <nav2_mppi.yaml>      # prints the path of the file written

Why each one:
  FollowPath.vx_max 0.25        the mission node's envelope caps the speed there (safety.max_linear_vel)
  inflation_radius 0.5, cost_scaling_factor 5.0
                                1.0 / 3.0 is for pavements: in a 2 m corridor every cell is then inside
                                some wall's inflation, and beside a plant pot (1.4 m left to pass) MPPI
                                asked 0.00-0.03 m/s and turned on the spot until the mission called it
                                stuck (ROS e2e, 12-oct: 2 and 4 times in 7 minutes)
"""
import hashlib
import os
import sys
import tempfile

import yaml

INDOOR = {
    ('controller_server', 'ros__parameters', 'FollowPath', 'vx_max'): 0.25,
    ('local_costmap', 'local_costmap', 'ros__parameters', 'inflation_layer', 'inflation_radius'): 0.5,
    ('local_costmap', 'local_costmap', 'ros__parameters', 'inflation_layer', 'cost_scaling_factor'): 5.0,
}


def indoor_params_file(nav2_yaml, overrides=None):
    with open(nav2_yaml) as fh:
        cfg = yaml.safe_load(fh)
    for path, value in {**INDOOR, **(overrides or {})}.items():
        node = cfg
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    text = yaml.safe_dump(cfg)
    # one file per content, not one per launch: the same launch twice writes the same file
    out = os.path.join(tempfile.gettempdir(), f'erc_nav2_mppi_indoor_{hashlib.sha1(text.encode()).hexdigest()[:10]}.yaml')
    tmp = f'{out}.{os.getpid()}'
    with open(tmp, 'w') as fh:
        fh.write(text)
    os.replace(tmp, out)
    return out


if __name__ == '__main__':
    print(indoor_params_file(sys.argv[1]))
