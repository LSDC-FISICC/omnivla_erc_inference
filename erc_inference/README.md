# OmniVLA ERC Inference

ROS 2 Python package for running OmniVLA edge inference and checkpoint-based missions.

## Build

From the workspace root:

```bash
colcon build --packages-up-to erc_inference
source install/setup.bash
```

## Run

```bash
ros2 run erc_inference omnivla_edge_node
ros2 run erc_inference checkpoint_controller_node
```

The inference node expects a model checkpoint and the configured camera, GPS, compass, and goal topics. Install the model's Python dependencies and source any required OmniVLA environment before starting the node.

## Motion controller

`omnivla_edge_node` turns localization and the model's selected waypoint into `/omnivla/cmd_vel` with one of two control laws, chosen by `controller_type` in [`config/controller.yaml`](config/controller.yaml). The laws live in [`erc_inference/motion_control.py`](erc_inference/motion_control.py), free of ROS and torch.

- `polar` (default) — bearing tracking. `polar.steering_source` picks what it steers toward:
  - `plan` (default): the model is the local planner, bounded by A*. The rover steers at the carrot plus the plan's *residual*: the bearing of the point `polar.plan_lookahead_m` (1.5 m) along the model's predicted path, minus the share of the carrot's bearing the model turns by on its own (`polar.plan_bearing_gain`, 0.117 on mission_16sept), less a `polar.plan_deviation_deadzone_deg` (8°) deadzone, capped at `polar.max_plan_deviation_deg` (±30°). With nothing to go around, it tracks like the carrot alone; a deviation the model plans beyond its usual underturning reaches the rover. A path shorter than `polar.plan_min_length_m` leaves the carrot alone in charge.
  - `carrot`: the route goal alone, located by GPS + heading; the model is only logged.
  - `model`: the model's index-4 waypoint, unbounded — mission_16sept's steering, where the model asked for about a ninth of the turn the route needed.

  `polar.linear_law` picks the speed: `cruise` (default, 0.3 m/s tapering to `polar.min_linear_vel` on large bearings, never a speed below that floor) or `rho` (Siegwart's `k_rho` × waypoint distance, which fell to 0.05 m/s and left the rover standing on mission_16sept).
- `pid` — heading PID on the bearing to the model's waypoint only.

`checkpoint_controller_node` then limits acceleration on `/cmd_vel` (`max_linear_accel`, `max_linear_decel`, `max_angular_accel`, `max_angular_decel`); checkpoint stops and cancels stay immediate.

**Timing.** The control law runs on the edge node's 3 Hz timer with the latest value of each input; it never waits for a new GPS fix or camera frame. `checkpoint_controller_node` republishes the latest command on `/cmd_vel` at 10 Hz. On `mission_16sept` the inputs behind each tick were old: heading content ~1.10 s, newest GPS fix ~1.04 s, and `/erc/gps/filtered` extrapolated from ~1.1 s-old speed and heading. A command's effect took ~1.3 s to appear in that telemetry. `polar.delay_compensation_s` / `_gain` subtract the turn still in transit over those 1.3 s. The simulator passes with the telemetry delay modelled either as real latency or as a rover clock offset (`RoverModel.telemetry_latency_s`); without the compensation the yaw rate changes sign ~3× as often. Detail: `Earth-rover-ros2-bridge/docs/ARQUITECTURA_ACTUAL.md` §8.4.

`scripts/omni_vla_wrapper` loads that file automatically (override it with `OMNIVLA_CONTROLLER_CONFIG=/path/to.yaml`). When running the node directly, pass it yourself:

```bash
ros2 run erc_inference omnivla_edge_node --ros-args \
  --params-file $(ros2 pkg prefix erc_inference)/share/erc_inference/config/controller.yaml
```

Everything in it can be changed while the rover drives:

```bash
ros2 param set /omnivla_edge_node polar.steering_source model
ros2 param set /omnivla_edge_node polar.min_linear_vel 0.2
```

When the goal is more than `goal_turn.enter_deg` behind the rover, the node overrides either controller, stops for `goal_turn.brake_s`, and turns toward it in place. It uses the bearing from localization rather than the model, whose waypoint cannot point backward.

### Checking a controller change

[`test/controller_sim.py`](test/controller_sim.py) drives whole checkpoint missions in closed loop with this package's own control code and `config/controller.yaml`, against a rover model fitted to mission_16sept (delay, gains, breakaway speeds, localization error) and swept over the ranges those numbers are uncertain in. Run it after changing any gain:

```bash
source /opt/ros/jazzy/setup.bash && source ~/lsdc_ws/install/setup.bash
cd omnivla_erc_inference/erc_inference
python3 test/controller_sim.py          # needs utm and PyYAML (the project venv has both)
python3 -m pytest test/test_motion_control.py test/test_controller_sim.py   # also needs pytest
```

## Route following

`checkpoint_controller_node` plans each leg with A\* and drives it with a carrot: a goal `carrot_distance_m` (default 1.5 m) ahead of the rover's projection on the route, re-published at `carrot_rate_hz`, with the route's direction as `/goal_compass`. A checkpoint counts as reached within `checkpoint_proximity_m` (default 6 m); if the SDK rejects it, the radius halves, down to `min_checkpoint_proximity_m`, and the rover closes in.

Both nodes read position from `/erc/gps/filtered`, so `erc_localization`'s `localization_global.launch.py` must be running.