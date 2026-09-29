"""Track-frame geometry for indoor missions, where there is no GPS.

Free of ROS so that tests and the simulator can use it as is.

The track frame is a flat metric frame fixed to the venue: x/y in meters, yaw
counter-clockwise from +x (ENU convention), defined by the route file (see
config/indoor_nyu_track.yaml). The rover's pose in it is the local EKF's
odom pose (wheel speed + gyro yaw, erc_localization ekf_local) re-anchored at
mission start:

    T_track_base = T_track_odom * T_odom_base,   T_track_odom fixed per anchor

The inference nodes and the motion controller only take positions as
NavSatFix and heading as a compass bearing. They are fed a *pseudo-fix*: track
x/y placed as east/north around a fixed datum, and a compass heading of
90 - yaw. Nothing indoors is geo-referenced; the lat/lon only carries the
metric offset the nodes turn back into meters.

The datum sits on a UTM central meridian (-75, zone 18). omnivla_edge_node takes
the UTM delta between the pseudo-fix and the goal as meters. On the central
meridian the grid convergence is zero, so grid north equals the pseudo-north
used here, and the grid scale is 0.9996: a 1.5 m carrot comes out 0.6 mm
short. An arbitrary datum would rotate every goal by the convergence (about 0.7
deg at NYU's real longitude) for no benefit.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import yaml

DATUM_LAT = 40.0
DATUM_LON = -75.0
# WGS84 meridian (M) and prime-vertical (N) radii at the datum, which is what UTM
# is built on. A spherical 6378137 m for both made UTM read north 0.3% short and
# east 0.1% long of the track, bending diagonal goals by 0.1 deg
# (test_indoor_track.py::test_edge_node_arithmetic_recovers_track_offset).
_A, _E2 = 6378137.0, 6.69437999014e-3
_S2 = math.sin(math.radians(DATUM_LAT)) ** 2
_M_PER_DEG_LAT = math.radians(1.0) * _A * (1.0 - _E2) / (1.0 - _E2 * _S2) ** 1.5
_M_PER_DEG_LON = math.radians(1.0) * _A / math.sqrt(1.0 - _E2 * _S2) * math.cos(math.radians(DATUM_LAT))


def track_to_latlon(x: float, y: float) -> Tuple[float, float]:
    return DATUM_LAT + y / _M_PER_DEG_LAT, DATUM_LON + x / _M_PER_DEG_LON


def latlon_to_track(lat: float, lon: float) -> Tuple[float, float]:
    return (lon - DATUM_LON) * _M_PER_DEG_LON, (lat - DATUM_LAT) * _M_PER_DEG_LAT


def yaw_to_compass_deg(yaw_rad: float) -> float:
    """ENU yaw (0 = +x, CCW) -> the SDK compass convention (0 = +y, clockwise)
    that /erc/heading_deg and the inference nodes use."""
    return (90.0 - math.degrees(yaw_rad)) % 360.0


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def compose(self, other: 'Pose2D') -> 'Pose2D':
        """self * other: `other` expressed in self's frame, returned in self's parent."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(self.x + c * other.x - s * other.y,
                      self.y + s * other.x + c * other.y,
                      wrap(self.yaw + other.yaw))

    def inverse(self) -> 'Pose2D':
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(-c * self.x - s * self.y, s * self.x - c * self.y, wrap(-self.yaw))


def anchor(track_pose: Pose2D, odom_pose: Pose2D) -> Pose2D:
    """T_track_odom such that T_track_odom * odom_pose == track_pose."""
    return track_pose.compose(odom_pose.inverse())


@dataclass(frozen=True)
class IndoorCheckpoint:
    sequence: int
    name: str
    x: float
    y: float
    cone: str = ''
    prompt: str = ''
    via: Tuple[Tuple[float, float], ...] = ()


@dataclass
class IndoorTrack:
    name: str
    start: Pose2D
    checkpoints: List[IndoorCheckpoint] = field(default_factory=list)

    def checkpoint(self, sequence: int) -> Optional[IndoorCheckpoint]:
        return next((c for c in self.checkpoints if c.sequence == sequence), None)

    def leg_route(self, checkpoint: IndoorCheckpoint, x: float, y: float) -> List[Tuple[float, float]]:
        """From (x, y) through the checkpoint's `via` corners to the checkpoint."""
        return [(x, y)] + list(checkpoint.via) + [(checkpoint.x, checkpoint.y)]

    def departure_pose(self, sequence: int) -> Pose2D:
        """Where a rover resuming after `sequence` is assumed to stand: on that
        checkpoint, facing the first segment of the next leg. sequence 0 is the start."""
        if sequence <= 0:
            return self.start
        here = self.checkpoint(sequence)
        if here is None:
            raise ValueError(f'no checkpoint with sequence {sequence} in track {self.name!r}')
        later = [c for c in self.checkpoints if c.sequence > sequence]
        if not later:
            return Pose2D(here.x, here.y, self.start.yaw)
        nxt = min(later, key=lambda c: c.sequence)
        tx, ty = nxt.via[0] if nxt.via else (nxt.x, nxt.y)
        return Pose2D(here.x, here.y, math.atan2(ty - here.y, tx - here.x))


def load_track(path: str) -> IndoorTrack:
    with open(path) as f:
        doc = yaml.safe_load(f)
    s = doc.get('start', {})
    start = Pose2D(float(s.get('x', 0.0)), float(s.get('y', 0.0)), math.radians(float(s.get('yaw_deg', 0.0))))
    checkpoints = []
    for item in doc['checkpoints']:
        checkpoints.append(IndoorCheckpoint(
            sequence=int(item['sequence']),
            name=str(item.get('name', f"CP{item['sequence']}")),
            x=float(item['x']),
            y=float(item['y']),
            cone=str(item.get('cone', '')),
            prompt=str(item.get('prompt', '')),
            via=tuple((float(p[0]), float(p[1])) for p in item.get('via', []) or []),
        ))
    checkpoints.sort(key=lambda c: c.sequence)
    if len({c.sequence for c in checkpoints}) != len(checkpoints):
        raise ValueError(f'{path}: duplicate checkpoint sequence')
    return IndoorTrack(name=str(doc.get('name', path)), start=start, checkpoints=checkpoints)
