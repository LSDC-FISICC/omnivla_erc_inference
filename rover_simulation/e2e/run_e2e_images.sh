#!/bin/bash
# End-to-end in ROS for the image-goal mission: the REAL image_checkpoint_controller_node and
# carrot_controller_node from this source tree (and erc_perception's cones.py from its source
# tree), against fake_image_world.py (plant, NYU walls/tables/chairs/doors, cones, lagged
# free space and cone detections, fake SDK). Parameters as mission_indoor.launch.py sets them.
#   ./run_e2e_images.sh <seconds> <seed> [layout loop|random] [world tables|chairs|chairs+doors] [tag]
# env: PLANT_KW_MOVING WHEEL_SCALE GYRO_SCALE GYRO_BIAS_DEG_MIN IMAGE_LAG_S SDK_PORT ROS_DOMAIN_ID_E2E
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../../erc_inference" && pwd)
PERC=${PERC:-$(cd "$HERE/../../../erc_perception" 2>/dev/null && pwd)}
source /opt/ros/jazzy/setup.bash
source ${WS:-$HOME/lsdc_ws}/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_E2E:-83} PYTHONPATH=$SRC:$PERC:${EXTRA_PYTHONPATH:-}:$PYTHONPATH
D=${1:-1800}; SEED=${2:-0}; LAYOUT=${3:-loop}; WORLD=${4:-chairs}; TAG=${5:-images_${LAYOUT}_${WORLD}_s${SEED}}
export SDK_PORT=${SDK_PORT:-8766}
mkdir -p "$HERE/out"; OUT="$HERE/out/${TAG}"
python3 "$HERE/fake_image_world.py" "$D" "$SEED" "$LAYOUT" "$WORLD" > "$OUT.world.log" 2>&1 & FW=$!
sleep 2
python3 -m erc_inference.carrot_controller_node --ros-args --params-file "$SRC/config/controller.yaml" \
  -p gps_topic:=/erc/indoor/fix -p compass_topic:=/erc/indoor/heading_deg \
  -p max_linear_vel:=0.25 -p goal_turn.enter_deg:=45.0 -p goal_turn.exit_deg:=15.0 > "$OUT.carrot.log" 2>&1 & CN=$!
python3 -m erc_inference.image_checkpoint_controller_node --ros-args \
  -p route_file:="$SRC/config/indoor_nyu_track.yaml" -p goals_file:="$SRC/config/indoor_nyu_goals.yaml" \
  -p checkpoint_reached_url:=http://127.0.0.1:$SDK_PORT/checkpoint-reached \
  -p checkpoint_list_url:=http://127.0.0.1:$SDK_PORT/checkpoints-list \
  -p safety.stop_if_attitude_stale:=false > "$OUT.mission.log" 2>&1 & MN=$!
sleep 5
ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \
  "{resume_from_latest_scanned: false}" > "$OUT.action.log" 2>&1 &
wait $FW
kill -INT $CN $MN 2>/dev/null; sleep 1; kill $CN $MN 2>/dev/null
pkill -f "action send_goal /start_mission" 2>/dev/null
cat "$OUT.world.log"
