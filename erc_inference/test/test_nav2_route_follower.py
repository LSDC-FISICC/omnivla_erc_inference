"""nav2_route_follower_node's pure functions: the online yaw-gain estimate and the command snap."""
import numpy as np

from erc_inference.nav2_route_follower_node import estimate_k_w, snap_command


def _replay(k_moving, k_in_place, delay=1.3, step=0.1, seed=0):
    """MPPI-like commands (a new one every `step` s, slowly varying) through a delayed plant."""
    rng = np.random.default_rng(seed)
    cmds = [(t, 0.25 if (t // 6) % 2 == 0 else 0.0, 0.25 * np.sin(t / 3.0) + rng.normal(0, 0.01))
            for t in np.arange(0, 40, step)]
    ct = np.array([c[0] for c in cmds])
    rates = []
    for t in np.arange(2, 40, 0.05):
        _t, v, w = cmds[int(np.searchsorted(ct, t - delay, side='right')) - 1]
        rates.append((t, (k_moving if v > 0 else k_in_place) * w + rng.normal(0, 0.03)))
    return cmds, rates


def test_learns_the_wuhan_gain_from_mppi_like_commands():
    cmds, rates = _replay(1.23, 0.93)
    k = estimate_k_w(cmds, rates, 40.0, 1.3, 30.0, 0.2)
    assert abs(k['moving'][0] - 1.23) < 0.08 and k['moving'][1] > 50
    assert abs(k['in_place'][0] - 0.93) < 0.08


def test_learns_the_16sept_gain():
    cmds, rates = _replay(0.36, 1.18, seed=1)
    k = estimate_k_w(cmds, rates, 40.0, 1.3, 30.0, 0.2)
    assert abs(k['moving'][0] - 0.36) < 0.05
    assert abs(k['in_place'][0] - 1.18) < 0.08


def test_no_turns_no_estimate():
    cmds = [(t, 0.25, 0.0) for t in np.arange(0, 20, 0.1)]
    rates = [(t, 0.0) for t in np.arange(2, 20, 0.05)]
    k = estimate_k_w(cmds, rates, 20.0, 1.3, 15.0, 0.2)
    assert k['moving'][1] == 0 and k['in_place'][1] == 0


def test_snap_divides_by_the_gain_and_keeps_the_floor():
    v, w = snap_command(0.1, 0.1, k_w_moving=1.25, w_max=0.3)
    assert v == 0.25 and abs(w - 0.1 * 0.25 / 0.1 / 1.25) < 1e-9
    assert snap_command(0.0, 0.05, w_floor=0.15) == (0.0, 0.15)
