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