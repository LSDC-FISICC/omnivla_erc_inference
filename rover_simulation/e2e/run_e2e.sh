#!/bin/bash
# End-to-end in ROS with the REAL nodes (checkpoint_controller_node, carrot_controller_node)
# from this source tree, a fake rover (measured plant + obstacles.py profile) and a fake SDK.
#   ./run_e2e.sh <hedge|wall|chicane> <seconds> <local_replan true|false> [carrot_m]
# Logs go to ./out/. Needs a workspace with erc_inference_msgs / erc_static_map_msgs.
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../../erc_inference" && pwd)
source /opt/ros/jazzy/setup.bash
source ${WS:-$HOME/lsdc_ws}/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_E2E:-78} PYTHONPATH=$SRC:$PYTHONPATH
W=$1; D=${2:-300}; LR=${3:-true}; C=${4:-1.5}
mkdir -p "$HERE/out"; OUT="$HERE/out/${W}_lr${LR}_c${C}"
python3 "$HERE/fake_world.py" "$W" "$D" > "$OUT.world.log" 2>&1 & FW=$!
sleep 2
python3 -m erc_inference.carrot_controller_node --ros-args --params-file "$SRC/config/controller.yaml" \
  -p obstacle_sidestep:=true > "$OUT.carrot.log" 2>&1 & CN=$!
python3 -m erc_inference.checkpoint_controller_node --ros-args \
  -p checkpoint_list_url:=http://127.0.0.1:8765/checkpoints-list \
  -p checkpoint_reached_url:=http://127.0.0.1:8765/checkpoint-reached \
  -p costmap_service_timeout_s:=1.0 -p planner_service_timeout_s:=1.0 \
  -p local_replan:=$LR -p carrot_distance_m:=$C > "$OUT.cp.log" 2>&1 & CP=$!
sleep 4
ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \
  "{resume_from_latest_scanned: false}" > "$OUT.action.log" 2>&1 &
wait $FW
kill -INT $CN $CP 2>/dev/null; sleep 1; kill $CN $CP 2>/dev/null
# a send_goal whose mission never finished waits forever and would start the next run
pkill -f "action send_goal /start_mission" 2>/dev/null
cat "$OUT.world.log"
