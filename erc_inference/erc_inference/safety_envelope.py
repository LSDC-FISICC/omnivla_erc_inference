"""Last limits on the command before it reaches the rover: speed, tilt, what is ahead.

Free of ROS, like motion_control.py. The node that owns /cmd_vel applies it
after the control law and before CommandShaper, so it holds whatever produced
the command (either model node, carrot_controller_node, a `ros2 param set` gone
wrong). controller.yaml's max_linear_vel/max_angular_vel only bound the control
law; nothing downstream enforced them, and the bridge accepts up to 1 m/s and
1 rad/s (erc_control_node normalises by 1.0 and clamps).

Three layers, each answering a different way to hurt the rover:

1. Caps: linear in [-max_reverse_vel, max_linear_vel], |angular| <=
   max_angular_vel, and |linear * angular| <= max_lateral_accel. The last one
   slows the rover in a turn instead of weakening the turn. A speed that ends up
   below min_moving_linear_vel becomes 0, i.e. a turn in place: the rover does
   not move below ~0.25 m/s anyway (mission_16sept), so a lower value is not
   "slower", it is a command nothing executes.
   At 0.3 m/s and 0.3 rad/s the lateral acceleration is 0.09 m/s^2, far from
   tipping a rover this low. Mission_16sept's rollover was a planter climbed at
   cruise speed, which is what layers 2 and 3 are for.
2. Tilt: roll/pitch from the attitude filter (imu_filter_madgwick,
   /erc/imu_attitude), measured as the angle between gravity now and gravity
   when the envelope was armed (mission start, rover standing on the floor).
   Relative, because how the IMU sits in the body is not settled on this rover
   (heading.yaml's axis-map notes). Above max_tilt_deg: full stop, latched until
   the tilt is back under tilt_resume_deg for tilt_resume_s. Climbing something
   shows up here first. The telemetry is ~1.1 s old, so this stops a climb in
   progress, not the first contact.
3. Obstacle ahead (only with a free-space profile, erc_perception's
   /erc/free_space): if the closest range within +-stop_sector_deg is below
   stop_distance_m on stop_confirm_ticks consecutive profiles, linear goes to 0
   and angular is kept, so the rover can still turn away. That is what a corner's
   end wall needs. It resumes after clear_confirm_ticks clear profiles.
   stop_distance_m 1.2: covers the 1.3 s command delay at 0.3 m/s plus braking;
   0.8 m hit in simulation (mission_carrot.launch.py). At 0.25 m/s the delay plus braking
   covers ~0.4 m, and indoors (2 m corridors) 1.2 m blocks every oblique heading: use
   0.7 there (image_checkpoint_controller_node's defaults). If the turn asked for is too
   small to move the rover, it turns in place toward the open side (unblock_turn_w).

limit() returns (linear, angular, hard, note). hard=True means stop NOW: the
caller bypasses the acceleration limits, as it does for checkpoint stops.
"""

import math
from dataclasses import dataclass

import numpy as np
from typing import Optional, Sequence, Tuple

DEFAULTS = {
    'safety.max_linear_vel': 0.3,
    'safety.max_reverse_vel': 0.0,
    'safety.max_angular_vel': 0.3,
    'safety.max_lateral_accel': 0.08,
    'safety.min_moving_linear_vel': 0.25,
    'safety.tilt_enabled': True,
    'safety.max_tilt_deg': 15.0,
    'safety.tilt_resume_deg': 8.0,
    'safety.tilt_resume_s': 1.0,
    'safety.attitude_timeout_s': 1.5,
    'safety.stop_if_attitude_stale': True,
    'safety.obstacle_enabled': False,
    'safety.stop_distance_m': 1.2,
    'safety.stop_sector_deg': 20.0,
    'safety.stop_confirm_ticks': 2,
    'safety.clear_confirm_ticks': 3,
    'safety.free_space_timeout_s': 1.5,
    'safety.stop_if_perception_stale': True,
    # Blocked with a turn too small to move the rover (below ~0.15 rad/s it does not turn in
    # place): turn in place toward the more open side instead, until the sector clears. Without
    # it the rover waits forever in front of a cone it has just reached (test/image_goal_sim.py).
    'safety.unblock_turn_w': 0.3,
    'safety.min_turn_w': 0.15,
}


def parameter_errors(p) -> list:
    errors = []
    for name in ('safety.max_linear_vel', 'safety.max_reverse_vel', 'safety.max_angular_vel',
                 'safety.max_lateral_accel', 'safety.min_moving_linear_vel'):
        if p[name] < 0.0:
            errors.append(f'{name} must be >= 0')
    if p['safety.tilt_resume_deg'] > p['safety.max_tilt_deg']:
        errors.append('safety.tilt_resume_deg must be <= safety.max_tilt_deg')
    if p['safety.stop_confirm_ticks'] < 1 or p['safety.clear_confirm_ticks'] < 1:
        errors.append('safety.*_confirm_ticks must be >= 1')
    return errors


def up_in_body(qx, qy, qz, qw) -> Tuple[float, float, float]:
    """World +z expressed in the body frame, for an orientation quaternion (body -> world)."""
    # third row of R(q): R^T * [0, 0, 1]
    return (2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))


def angle_between_deg(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (na * nb)))))


def cap(linear: float, angular: float, p) -> Tuple[float, float, str]:
    """Layer 1 alone: the speed, turn-rate and lateral-acceleration caps."""
    notes = []
    lin = min(max(linear, -p['safety.max_reverse_vel']), p['safety.max_linear_vel'])
    ang = max(-p['safety.max_angular_vel'], min(p['safety.max_angular_vel'], angular))
    if lin != linear or ang != angular:
        notes.append('cap')
    a_max = p['safety.max_lateral_accel']
    if abs(ang) > 1e-6 and abs(lin * ang) > a_max:
        lin = math.copysign(a_max / abs(ang), lin)
        notes.append('lat-accel')
    if 0.0 < abs(lin) < p['safety.min_moving_linear_vel']:
        lin = 0.0
        notes.append('below-floor->turn-in-place')
    return lin, ang, ','.join(notes)


@dataclass
class _Attitude:
    up: Tuple[float, float, float]
    t: float


class SafetyEnvelope:
    def __init__(self):
        self._ref_up: Optional[Tuple[float, float, float]] = None
        self._att: Optional[_Attitude] = None
        self._tilted = False
        self._level_since: Optional[float] = None
        self.tilt_deg = 0.0
        self._scan = None               # (t, bearings_deg, ranges, range_max)
        self._scan_seen = None          # t of the last profile counted
        self._blocked_n = 0
        self._clear_n = 0
        self._obstacle_stop = False

    # -- inputs -------------------------------------------------------------

    def arm(self):
        """Take the current attitude as level. Call with the rover standing on the floor."""
        self._ref_up = self._att.up if self._att is not None else None
        self._tilted = False
        self._level_since = None

    def observe_attitude(self, t, qx, qy, qz, qw):
        self._att = _Attitude(up_in_body(qx, qy, qz, qw), t)
        if self._ref_up is None:
            self._ref_up = self._att.up

    def observe_free_space(self, t, bearings_deg: Sequence[float], ranges: Sequence[float], range_max: float):
        self._scan = (t, list(bearings_deg), list(ranges), float(range_max))

    # -- layers 2 and 3 -----------------------------------------------------

    def _tilt_gate(self, now, p):
        if not p['safety.tilt_enabled']:
            return False, ''
        if self._att is None or now - self._att.t > p['safety.attitude_timeout_s']:
            if p['safety.stop_if_attitude_stale']:
                return True, 'attitude stale'
            return False, 'attitude stale, ignored'
        self.tilt_deg = angle_between_deg(self._att.up, self._ref_up)
        if not self._tilted and self.tilt_deg > p['safety.max_tilt_deg']:
            self._tilted, self._level_since = True, None
        elif self._tilted:
            if self.tilt_deg < p['safety.tilt_resume_deg']:
                self._level_since = now if self._level_since is None else self._level_since
                if now - self._level_since >= p['safety.tilt_resume_s']:
                    self._tilted = False
            else:
                self._level_since = None
        return self._tilted, f'tilt {self.tilt_deg:.1f} deg' if self._tilted else ''

    @property
    def tilted(self):
        return self._tilted

    def ahead_m(self, p) -> Optional[float]:
        if self._scan is None:
            return None
        _t, bearings, ranges, range_max = self._scan
        half = p['safety.stop_sector_deg']
        inside = [r for b, r in zip(bearings, ranges) if abs(b) <= half and math.isfinite(r)]
        return min(inside) if inside else range_max

    def _obstacle_gate(self, now, p):
        if not p['safety.obstacle_enabled']:
            return False, ''
        if self._scan is None or now - self._scan[0] > p['safety.free_space_timeout_s']:
            if p['safety.stop_if_perception_stale']:
                return True, 'perception stale'
            return False, 'perception stale, ignored'
        if self._scan_seen != self._scan[0]:          # count each profile once
            self._scan_seen = self._scan[0]
            ahead = self.ahead_m(p)
            if ahead < p['safety.stop_distance_m']:
                self._blocked_n, self._clear_n = self._blocked_n + 1, 0
            else:
                self._blocked_n, self._clear_n = 0, self._clear_n + 1
            if not self._obstacle_stop and self._blocked_n >= p['safety.stop_confirm_ticks']:
                self._obstacle_stop = True
            elif self._obstacle_stop and self._clear_n >= p['safety.clear_confirm_ticks']:
                self._obstacle_stop = False
        return self._obstacle_stop, f'obstacle {self.ahead_m(p):.2f} m ahead' if self._obstacle_stop else ''

    # -- all together -------------------------------------------------------

    def limit(self, now: float, linear: float, angular: float, p) -> Tuple[float, float, bool, str]:
        tilted, tilt_note = self._tilt_gate(now, p)
        if tilted:
            # the one thing allowed while tilted: straight back, down off whatever it climbed
            if linear < 0.0 and abs(angular) < 1e-6:
                return max(linear, -p['safety.max_reverse_vel']), 0.0, False, f'[safety] tilted: backing off ({tilt_note})'
            return 0.0, 0.0, True, f'[safety] STOP {tilt_note}'
        lin, ang, cap_note = cap(linear, angular, p)
        blocked, obs_note = self._obstacle_gate(now, p)
        hard = False
        if blocked and lin > 0.0:
            lin, hard = 0.0, True
            if abs(ang) < p['safety.min_turn_w'] and p['safety.unblock_turn_w'] > 0.0 and self._scan is not None \
                    and self._obstacle_stop:           # blocked by something seen, not by stale perception
                _t, b, r, _cap = self._scan
                left = [rr for bb, rr in zip(b, r) if bb > p['safety.stop_sector_deg']]
                right = [rr for bb, rr in zip(b, r) if bb < -p['safety.stop_sector_deg']]
                side = 1.0 if (np.mean(left) if left else 0.0) >= (np.mean(right) if right else 0.0) else -1.0
                ang = side * p['safety.unblock_turn_w']
                obs_note += ', turning to unblock'
        notes = [n for n in (cap_note, obs_note, tilt_note) if n]
        return lin, ang, hard, ('[safety] ' + '; '.join(notes)) if notes else ''
