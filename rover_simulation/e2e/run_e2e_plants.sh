#!/bin/bash
# End-to-end in ROS for the potted-plant search: the REAL image_checkpoint_controller_node,
# started by launch/plants_controller.launch.py (explore:=true, /erc/objects), and
# carrot_controller_node from this source tree, against fake_plants_world.py (plant, a floor the
# mission does not know, plants, lagged free space and detections). Carrot parameters as
# mission_plants.launch.py sets them.
#   ./run_e2e_plants.sh <seconds> <seed> [plants to show] [tag]
# env: PLANT (unit|competition) KW_MOVING_SCALE WHEEL_SCALE GYRO_SCALE GYRO_BIAS_DEG_MIN IMAGE_LAG_S
#      ROS_DOMAIN_ID_E2E EXTRA_CARROT_ARGS (more -p name:=value for carrot_controller_node)
#      EXTRA_LAUNCH_ARGS (more name:=value for plants_controller.launch.py)
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../../erc_inference" && pwd)
source /opt/ros/jazzy/setup.bash
source ${WS:-$HOME/lsdc_ws}/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_E2E:-84} PYTHONPATH=$SRC:${EXTRA_PYTHONPATH:-}:$PYTHONPATH
D=${1:-900}; SEED=${2:-0}; N=${3:-3}; TAG=${4:-plants_s${SEED}_n${N}}
mkdir -p "$HERE/out"; OUT="$HERE/out/${TAG}"
python3 "$HERE/fake_plants_world.py" "$D" "$SEED" "$N" > "$OUT.world.log" 2>&1 & FW=$!
sleep 2
python3 -m erc_inference.carrot_controller_node --ros-args --params-file "$SRC/config/controller.yaml" \
  -p gps_topic:=/erc/indoor/fix -p compass_topic:=/erc/indoor/heading_deg \
  -p max_linear_vel:=0.25 -p goal_turn.enter_deg:=30.0 -p goal_turn.exit_deg:=10.0 ${EXTRA_CARROT_ARGS:-} > "$OUT.carrot.log" 2>&1 & CN=$!
# In its own session with SIGINT at its default, as in a terminal: a background job of a script
# has SIGINT ignored, python keeps it ignored, and the launch then outlives this script with the
# node still serving /start_mission (the next run got two missions driving at once, 9-oct).
setsid python3 -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])' \
  ros2 launch "$SRC/launch/plants_controller.launch.py" plants:=$N ${EXTRA_LAUNCH_ARGS:-} > "$OUT.mission.log" 2>&1 & MN=$!
sleep 6
ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \
  "{resume_from_latest_scanned: false}" > "$OUT.action.log" 2>&1 & AG=$!
wait $FW
sleep 3                      # let the action report its result
kill -INT -- -$MN 2>/dev/null          # Ctrl-C in the controller's terminal: the whole group
for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -g $MN > /dev/null || break; sleep 1; done
if pgrep -g $MN > /dev/null; then echo "Ctrl-C: controller STILL ALIVE after 10 s"; kill -KILL -- -$MN; else echo "Ctrl-C: controller gone after ${i} s"; fi
kill $CN $AG 2>/dev/null
cat "$OUT.world.log"
echo "--- action"; tail -6 "$OUT.action.log"
