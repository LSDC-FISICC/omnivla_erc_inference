#!/bin/bash
# Copy only the topics needed to review a mission visually (no camera images) into a
# small MCAP bag, e.g. to open in Foxglove on a laptop without ROS.
#   ./extract_viz.sh <bag> [<out dir>]      (default out: <bag>_viz)
# Add /erc/front_camera to TOPICS if you want the video too (that is most of the size).
set -e
IN=$1; OUT=${2:-${IN%/}_viz}
TOPICS="/erc/global_route, /erc/local_route, /erc/local_costmap, /erc/carrot, /erc/free_space_leg,
        /erc/free_space, /tf, /tf_static, /erc_static_map/costmap, /erc_static_map/planned_path,
        /erc/odometry/global, /omnivla_debug, /erc/free_space_debug, /rosout"
source /opt/ros/jazzy/setup.bash
CFG=$(mktemp --suffix=.yaml)
printf 'output_bags:\n  - uri: %s\n    storage_id: mcap\n    topics: [%s]\n' "$OUT" "$(echo $TOPICS)" > "$CFG"
ros2 bag convert -i "$IN" -o "$CFG"
rm -f "$CFG"
ros2 bag info "$OUT" | grep -E "Bag size|Messages"
