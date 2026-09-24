#!/bin/bash
# End-to-end in ROS: Nav2 MPPI (controller_server + local_costmap from the DA3 profile)
# as the local layer, with the REAL checkpoint_controller_node and nav2_route_follower_node
# from this source tree, a fake rover (measured plant + obstacles.py profile) and a fake SDK.
#   ./run_e2e_nav2.sh <hedge|wall|chicane> <seconds> [local_replan true|false]
# Needs Nav2 installed (ros-jazzy-nav2-controller, -nav2-mppi-controller, ...). Logs in ./out/.
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../../erc_inference" && pwd)
source /opt/ros/jazzy/setup.bash
source ${WS:-$HOME/lsdc_ws}/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_E2E:-78} PYTHONPATH=$SRC:$PYTHONPATH
W=$1; D=${2:-300}; LR=${3:-false}
# `ros2 run` starts the executable as a CHILD process: killing the wrapper's PID leaves
# controller_server running, and the next run then has two MPPIs publishing on
# /nav2/cmd_vel (it happened: 7 of them). Kill by what they are, on exit and up front.
cleanup() {
  pkill -INT -f "config/nav2_mppi.yaml" 2>/dev/null
  pkill -INT -f "erc_inference.nav2_route_follower_node|erc_inference.checkpoint_controller_node" 2>/dev/null
  pkill -f "$HERE/fake_world.py" 2>/dev/null
  # a send_goal whose mission never finished waits forever, and would start the NEXT
  # run's checkpoint controller at a random moment (15 of them were found alive)
  pkill -f "action send_goal /start_mission" 2>/dev/null
  sleep 1
  pkill -9 -f "config/nav2_mppi.yaml" 2>/dev/null
  pkill -9 -f "erc_inference.nav2_route_follower_node|erc_inference.checkpoint_controller_node" 2>/dev/null
}
cleanup
trap cleanup EXIT
mkdir -p "$HERE/out"; OUT="$HERE/out/${W}_nav2_lr${LR}${TAG:+_$TAG}"
python3 "$HERE/fake_world.py" "$W" "$D" > "$OUT.world.log" 2>&1 & FW=$!
sleep 2
# MPPI_EXTRA: extra --ros-args overrides, e.g. the Ackermann variant:
#   MPPI_EXTRA="-p FollowPath.motion_model:=Ackermann -p FollowPath.AckermannConstraints.min_turning_r:=2.5"
ros2 run nav2_controller controller_server --ros-args --params-file "$SRC/config/nav2_mppi.yaml" \
  -r cmd_vel:=/nav2/cmd_vel $MPPI_EXTRA > "$OUT.controller.log" 2>&1 & CS=$!
# give controller_server time to be discovered: a configure request sent too early
# loses its response (DDS) and the server stays inactive forever
sleep 3
ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_mppi \
  --params-file "$SRC/config/nav2_mppi.yaml" > "$OUT.lifecycle.log" 2>&1 & LM=$!
python3 -m erc_inference.nav2_route_follower_node > "$OUT.follower.log" 2>&1 & RF=$!
python3 -m erc_inference.checkpoint_controller_node --ros-args \
  -p checkpoint_list_url:=http://127.0.0.1:8765/checkpoints-list \
  -p checkpoint_reached_url:=http://127.0.0.1:8765/checkpoint-reached \
  -p costmap_service_timeout_s:=1.0 -p planner_service_timeout_s:=1.0 \
  -p local_replan:=$LR -p carrot_distance_m:=1.5 -p local_costmap_rate_hz:=3.0 > "$OUT.cp.log" 2>&1 & CP=$!
sleep 6
ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \
  "{resume_from_latest_scanned: false}" > "$OUT.action.log" 2>&1 &
wait $FW
cat "$OUT.world.log"
