"""checkpoint_controller_node's reflexes outdoors (20-oct): slopes it must drive, a tilt it must not,
and people in the way. Builds the real node (no executor, no network) and drives _reflexes().
Run by hand: python3 -m pytest test/test_outdoor_reflexes.py"""
import json
import math
import time

import pytest

rclpy = pytest.importorskip('rclpy')
from std_msgs.msg import String  # noqa: E402

from erc_inference.checkpoint_controller_node import CheckpointControllerNode  # noqa: E402

G = 9.81


@pytest.fixture(scope='module')
def node():
    rclpy.init()
    n = CheckpointControllerNode()
    yield n
    n.destroy_node()
    rclpy.shutdown()


def pitched(node, t, deg):
    """The attitude filter and the accelerometer reading a pitch of deg."""
    a = math.radians(deg)
    node._envelope.observe_attitude(t, 0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
    node._envelope.observe_accel(t, G * math.sin(a), 0.0, G * math.cos(a))


def start(node, t0):
    node._mission_active = True
    pitched(node, t0 - 1.0, 0.0)
    pitched(node, t0, 0.0)
    node._envelope.arm()
    node._was_tilted, node._tilt_back_until = False, None
    node._people_wait.__init__(**node._people_wait.p)


def run(node, t0, seconds, deg, cmd=(0.3, 0.1), dt=0.1):
    out = []
    t = t0
    while t < t0 + seconds:
        pitched(node, t, deg)
        out.append((t, *node._reflexes(t, *cmd)[:2]))
        t += dt
    return out


def test_a_slope_of_18_deg_is_driven(node):
    t0 = time.monotonic()
    start(node, t0)
    out = run(node, t0, 3.0, 18.0)
    assert all(v == 0.3 and w == 0.1 for _t, v, w in out)


def test_a_tilt_past_25_deg_stops_backs_down_and_tries_again(node):
    t0 = time.monotonic() + 100.0
    start(node, t0)
    out = run(node, t0, 12.0, 30.0)
    back = [t for t, v, w in out if v < 0.0]
    assert back and all(w == 0.0 for _t, v, w in out if v < 0.0)          # straight back only
    assert not any(v > 0.0 for t, v, w in out if t > t0 + 0.15)             # never forward once tilted (the
    # accelerometer median confirms within a tick of an instant 0 -> 30 deg step)
    gaps = [b - a for a, b in zip(back[:-1], back[1:]) if b - a > 0.15]
    assert len(gaps) == node._param('tilt.tries') - 1                       # three back-offs, apart
    level = run(node, t0 + 12.0, 3.0, 15.0)                                 # back on a gentle slope
    assert level[-1][1:] == (0.3, 0.1)                                      # under resume_deg: on


def test_someone_in_the_way_holds_the_rover_until_gone_and_seen_again(node):
    t0 = time.monotonic() + 200.0
    start(node, t0)
    det = {'label': 'person', 'bearing_deg': 0.0, 'bearing_right_deg': -6.0, 'bearing_left_deg': 6.0,
           'range_m': 2.0}
    for k in range(3):                                                      # two sightings and more
        node._people_wait.observe(t0 + 0.2 * k, [(-6.0, 6.0, 2.0)])
    assert node._reflexes(t0 + 0.5, 0.3, 0.0)[:2] == (0.0, 0.0)
    for k in range(3):                                                      # gone: two profiles later, on
        node._people_wait.profile(t0 + 1.0 + 0.4 * k)
    assert node._reflexes(t0 + 2.2, 0.3, 0.0)[:2] == (0.3, 0.0)
    node._objects_callback(String(data=json.dumps({'stamp': 0.0, 'detections': [det]})))   # parses


def test_a_person_beside_the_path_does_not_hold(node):
    t0 = time.monotonic() + 300.0
    start(node, t0)
    for k in range(3):
        node._people_wait.observe(t0 + 0.2 * k, [(30.0, 40.0, 2.0)])       # 1.0-1.3 m to the left
    assert node._reflexes(t0 + 0.5, 0.3, 0.0)[:2] == (0.3, 0.0)
