#!/bin/bash
# End-to-end in ROS for the indoor stack: the REAL indoor_mission_node from this source
# tree, carrot_controller_node standing in for the model node (remapped to the indoor
# pseudo-fix and heading exactly as mission_indoor.launch.py remaps the model), and
# fake_indoor_world.py (measured plant, NYU walls, posed odometry).
#   ./run_e2e_indoor.sh <seconds> [tag]      env: PLANT_KW_MOVING WHEEL_SCALE GYRO_BIAS_DEG_MIN
# Logs go to ./out/. Needs a workspace with erc_inference_msgs (message packages only;
# the nodes run from source).
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../../erc_inference" && pwd)
source /opt/ros/jazzy/setup.bash
source ${WS:-$HOME/lsdc_ws}/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_E2E:-79} PYTHONPATH=$SRC:$PYTHONPATH
D=${1:-700}; TAG=${2:-indoor}
mkdir -p "$HERE/out"; OUT="$HERE/out/${TAG}"
python3 "$HERE/fake_indoor_world.py" "$D" > "$OUT.world.log" 2>&1 & FW=$!
sleep 2
python3 -m erc_inference.carrot_controller_node --ros-args --params-file "$SRC/config/controller.yaml" \
  -p gps_topic:=/erc/indoor/fix -p compass_topic:=/erc/indoor/heading_deg > "$OUT.carrot.log" 2>&1 & CN=$!
python3 -m erc_inference.indoor_mission_node --ros-args \
  -p route_file:="$SRC/config/indoor_nyu_track.yaml" -p confirm_with_sdk:=false > "$OUT.mission.log" 2>&1 & MN=$!
sleep 4
ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \
  "{resume_from_latest_scanned: false}" > "$OUT.action.log" 2>&1 &
wait $FW
kill -INT $CN $MN 2>/dev/null; sleep 1; kill $CN $MN 2>/dev/null
pkill -f "action send_goal /start_mission" 2>/dev/null
cat "$OUT.world.log"
