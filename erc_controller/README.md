# erc_controller

ROS 2 (Jazzy) package with the mission controllers of the LSDC-FISICC Earth Rover stack, outdoor
(GPS checkpoints over OpenStreetMap) and indoor (no GPS). Its messages are in `erc_controller_msgs`
(the `StartMission` action). Until 20-oct-2026 this was `omnivla_erc_inference` / `erc_inference`;
the OmniVLA model nodes were removed then.

The architecture, the field evidence behind each choice and the state machines are documented in
Earth-rover-ros2-bridge/docs: `ARQUITECTURA_ACTUAL.md`, `ARQUITECTURA_NAVEGACION.md`,
`ARQUITECTURA_PERCEPCION.md`. Operating guides: `COMANDOS_MISION.pdf` (outdoor),
`plants_mission_reactive.pdf` (indoor, plants), `offroad_mission.pdf`, `indoor_mission.pdf`.

## Build

```bash
cd ~/lsdc_ws && colcon build --packages-select erc_controller_msgs erc_controller
source install/setup.bash
```

## Nodes

| Node | Variant | What it does |
|---|---|---|
| `checkpoint_controller_node` | outdoor | GPS checkpoint mission: OSM A* per leg, local replanning, carrot, SDK confirmation; the only writer of `/cmd_vel`; tilt stop and people wait |
| `carrot_controller_node` | both | polar control on the carrot (`motion_control.MotionController`), side-step / obstacle stop |
| `nav2_route_follower_node` | both | glue to Nav2's MPPI (experimental) |
| `image_checkpoint_controller_node` | indoor | NYU cone patrol (`loop_patrol`) or, with `explore:=true`, the plant mission (`explore_search`, reactive drive) |
| `indoor_mission_node` | indoor | checkpoints from a track file, odometry only |
| `local_map_node` | indoor | view-only local map for Foxglove |
| `image_goal_offroad_node`, `flag_checkpoint_node` | off-road | photo homing, blue-flag checkpoints |

## Launch

| Launch | Use |
|---|---|
| `mission_carrot_unidepth.launch.py` (+ `mission_carrot`, `mission_mexico`) | outdoor, recommended |
| `mission_nav2.launch.py`, `mission_nav2_unidepth.launch.py` | outdoor, MPPI (experimental) |
| `mission_offroad.launch.py` | off-road |
| `mission_indoor.launch.py` | indoor NYU |
| `mission_plants.launch.py` + `plants_controller.launch.py` | indoor plants |

The mission node is always started by hand in its own terminal: killing it takes the rover back.

## Tests and simulation

`python3 -m pytest test/` (with ROS sourced). Closed-loop simulators: `test/controller_sim.py`
(outdoor control), `test/explore_search_sim.py` (plant mission), `rover_simulation/e2e/` (ROS end
to end with a fake rover and SDK). Change control only after running them.
