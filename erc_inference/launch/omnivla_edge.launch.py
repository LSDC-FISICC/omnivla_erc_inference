import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "omni_vla_edge_wrapper"
    return LaunchDescription([
        ExecuteProcess(
            cmd=[str(script_path)],
            output="screen",
        )
    ])
