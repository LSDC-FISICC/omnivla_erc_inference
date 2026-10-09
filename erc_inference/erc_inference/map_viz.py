"""A LocalCostmap and the objects on it as ROS messages, to look at (Foxglove, RViz, a bag).

Shared by image_checkpoint_controller_node (the exploring mission's own map) and
local_map_node (the same map built with nobody driving). Only the part of the grid that has
been seen is sent: the whole 60 x 60 m map is 90 kB a message, the seen part a few kB while the
rover has covered a room -- it goes through an SSH tunnel to whoever is watching.
"""
import math

import numpy as np
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray


def grid_message(cmap, stamp, frame, origin_pose=None, margin_cells=3):
    """cmap: local_planner.LocalCostmap. origin_pose: (x, y, yaw) of the map frame's origin in
    `frame` (None: the map frame IS `frame`). -> OccupancyGrid of the seen bounding box, or None
    while nothing has been seen. -1 unseen, 0-100 from the log-odds."""
    rows, cols = np.nonzero(cmap.seen)
    if len(rows) == 0:
        return None
    r0, r1 = max(int(rows.min()) - margin_cells, 0), min(int(rows.max()) + margin_cells + 1, cmap.h)
    c0, c1 = max(int(cols.min()) - margin_cells, 0), min(int(cols.max()) + margin_cells + 1, cmap.w)
    res = float(cmap.p['resolution_m'])
    prob = 100.0 / (1.0 + np.exp(-cmap.L[r0:r1, c0:c1]))
    data = np.where(cmap.seen[r0:r1, c0:c1], np.clip(prob, 0, 100), -1).astype(np.int8)
    ox, oy, oyaw = cmap.ox + c0 * res, cmap.oy + r0 * res, 0.0     # the crop's corner, map frame
    if origin_pose is not None:
        px, py, pyaw = origin_pose
        c, s = math.cos(pyaw), math.sin(pyaw)
        ox, oy, oyaw = px + c * ox - s * oy, py + s * ox + c * oy, pyaw
    g = OccupancyGrid()
    g.header.stamp, g.header.frame_id = stamp, frame
    g.info.resolution = res
    g.info.width, g.info.height = int(c1 - c0), int(r1 - r0)
    g.info.origin.position.x, g.info.origin.position.y = float(ox), float(oy)
    g.info.origin.orientation.z, g.info.origin.orientation.w = math.sin(oyaw / 2), math.cos(oyaw / 2)
    g.data = data.ravel().tolist()
    return g


def object_markers(objects, stamp, frame, origin_pose=None):
    """objects: [(class, x, y, visited, detections)] in the map frame -> MarkerArray: a cylinder
    and a label each; orange while pending, green once shown."""
    arr = MarkerArray()
    arr.markers.append(Marker(action=Marker.DELETEALL))
    for i, (cls, x, y, visited, n) in enumerate(objects):
        if origin_pose is not None:
            px, py, pyaw = origin_pose
            c, s = math.cos(pyaw), math.sin(pyaw)
            x, y = px + c * x - s * y, py + s * x + c * y
        colour = (0.2, 0.8, 0.2) if visited else (1.0, 0.6, 0.0)
        for k, kind in enumerate((Marker.CYLINDER, Marker.TEXT_VIEW_FACING)):
            mk = Marker(type=kind, action=Marker.ADD, ns='objects' if k == 0 else 'labels', id=i)
            mk.header.stamp, mk.header.frame_id = stamp, frame
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = float(x), float(y), 0.25 if k == 0 else 0.8
            mk.pose.orientation.w = 1.0
            mk.scale.x, mk.scale.y, mk.scale.z = (0.4, 0.4, 0.5) if k == 0 else (0.0, 0.0, 0.25)
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = (*colour, 0.9) if k == 0 else (1.0, 1.0, 1.0, 1.0)
            mk.text = (f'{cls}' + (' (shown)' if visited else '')) if k == 1 else ''
            arr.markers.append(mk)
    return arr
