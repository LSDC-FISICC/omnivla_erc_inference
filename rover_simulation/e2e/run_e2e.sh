#!/bin/bash
# End-to-end in ROS with the REAL nodes (checkpoint_controller_node, carrot_controller_node)
# from this source tree, a fake rover (measured plant + obstacles.py profile) and a fake SDK.
#   ./run_e2e.sh <world> <seconds> <local_replan true|false> [carrot_m]
# CP_EXTRA: extra checkpoint_controller_node args, e.g. blocked-route recovery:
#   CP_EXTRA="-p local.rejoin_respects_unseen:=true -p local.unseen_extend_m:=2.5 -p recovery.explore:=true"
# TAG: suffix for the output files.
# HEADING_SRC=gyro_gps|fused: the fake world publishes raw gyro/wheels/GPS/magnetometer instead of
#   the heading and Earth-rover-ros2-bridge's heading_node.py estimates it (see fake_world.py).
#   MAG_MODE=frozen|good: that magnetometer. MAG_CAL=valid: fused gets a valid calibration file.
# Logs go to ./out/. Needs a workspace with erc_inference_msgs / erc_static_map_msgs.
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../../erc_inference" && pwd)
source /opt/ros/jazzy/setup.bash
source ${WS:-$HOME/lsdc_ws}/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_E2E:-78} PYTHONPATH=$SRC:$PYTHONPATH
W=$1; D=${2:-300}; LR=${3:-true}; C=${4:-1.5}
mkdir -p "$HERE/out"; OUT="$HERE/out/${W}_lr${LR}_c${C}${TAG:+_$TAG}"
python3 "$HERE/fake_world.py" "$W" "$D" > "$OUT.world.log" 2>&1 & FW=$!
HN=
if [ "$HEADING_SRC" = gyro_gps ] || [ "$HEADING_SRC" = fused ]; then
  BRIDGE=${BRIDGE_SRC:-$(cd "$HERE/../../../Earth-rover-ros2-bridge" && pwd)}
  CAL="$OUT.magcal.yaml"; rm -f "$CAL"
  # MAG_CAL=valid: what mag_calibrate.py writes after a valid calibration of fake_world's
  # magnetometer (no hard iron); otherwise no file, as when it was never run
  [ "$MAG_CAL" = valid ] && printf "valid: true\nstamp: %s\ntime: 'e2e'\naxis_map: '-y,x,z'\nhard_iron: [0.0, 0.0, 0.0]\nsoft_iron: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n" "$(date +%s)" > "$CAL"
  python3 "$BRIDGE/erc_localization/scripts/heading_node.py" --ros-args -p source:=$HEADING_SRC \
    -p mag_calibration_file:="$CAL" > "$OUT.heading.log" 2>&1 & HN=$!
fi
sleep 2
python3 -m erc_inference.carrot_controller_node --ros-args --params-file "$SRC/config/controller.yaml" \
  -p obstacle_sidestep:=true > "$OUT.carrot.log" 2>&1 & CN=$!
python3 -m erc_inference.checkpoint_controller_node --ros-args \
  -p checkpoint_list_url:=http://127.0.0.1:8765/checkpoints-list \
  -p checkpoint_reached_url:=http://127.0.0.1:8765/checkpoint-reached \
  -p costmap_service_timeout_s:=1.0 -p planner_service_timeout_s:=1.0 \
  -p local_replan:=$LR -p carrot_distance_m:=$C $CP_EXTRA > "$OUT.cp.log" 2>&1 & CP=$!
sleep 4
ros2 action send_goal /start_mission erc_inference_msgs/action/StartMission \
  "{resume_from_latest_scanned: false}" > "$OUT.action.log" 2>&1 & AG=$!
wait $FW
kill -INT $CN $CP $HN 2>/dev/null; sleep 1; kill $CN $CP $HN 2>/dev/null
# a send_goal whose mission never finished waits forever and would start the next run. Its own
# PID only: a pkill -f pattern would also kill a real mission's send_goal on the same machine
kill $AG 2>/dev/null
cat "$OUT.world.log"
