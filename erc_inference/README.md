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

`omnivla_edge_node` turns the model's selected waypoint into `/cmd_vel` with one of two control laws, chosen by `controller_type` in [`config/controller.yaml`](config/controller.yaml):

- `polar` (default) — Siegwart's polar-coordinate controller. Steers on the bearing to the waypoint (and, with `polar.k_beta` < 0, on the heading the model predicts there).
- `pid` — heading PID on the bearing to the waypoint only.

`scripts/omni_vla_wrapper` loads that file automatically (override it with `OMNIVLA_CONTROLLER_CONFIG=/path/to.yaml`). When running the node directly, pass it yourself:

```bash
ros2 run erc_inference omnivla_edge_node --ros-args \
  --params-file $(ros2 pkg prefix erc_inference)/share/erc_inference/config/controller.yaml
```

Everything in it can be changed while the rover drives:

```bash
ros2 param set /omnivla_edge_node controller_type pid
ros2 param set /omnivla_edge_node polar.k_alpha 1.2
```

When the goal is more than `goal_turn.enter_deg` behind the rover, the node overrides either controller and turns toward it first (in place by default). It uses the bearing from localization rather than the model, whose waypoint cannot point backward.

## Route following

`checkpoint_controller_node` plans each leg with A\* and drives it with a carrot: a goal `carrot_distance_m` (default 1.5 m) ahead of the rover's projection on the route, re-published at `carrot_rate_hz`, with the route's direction as `/goal_compass`. A checkpoint counts as reached within `checkpoint_proximity_m` (default 3 m); if the SDK rejects it, the radius halves, down to `min_checkpoint_proximity_m`, and the rover closes in.

Both nodes read position from `/erc/gps/filtered`, so `erc_localization`'s `localization_global.launch.py` must be running.