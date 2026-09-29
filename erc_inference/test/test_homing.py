"""goal_homing + homing_mission: the off-road image-goal pieces. No ROS.

The geometry is checked on synthetic correspondences (known poses, the TAREA1 fisheye model),
the SIFT path on a textured image, the mission's decisions on scripted measurements, and the
whole loop once in test/homing_sim.py's arena.
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '..', '..', 'erc_perception'))

cv2 = pytest.importorskip('cv2')

from erc_inference.goal_homing import DEFAULTS, Camera, GoalMatcher, Homing  # noqa: E402
from erc_inference.homing_mission import ALIGN, ARRIVED, BACKOFF, MEASURE, PULSE, HomingGoal, HomingMission  # noqa: E402

CAM = Camera()


def geometry_only():
    m = GoalMatcher.__new__(GoalMatcher)
    m.cv2, m.cam, m.p, m.warning = cv2, CAM, dict(DEFAULTS), ''
    return m


def distort(xn, yn):
    import homing_sim
    return homing_sim._distort(xn, yn, CAM.lam)


def synth(fwd, left, yaw_deg, rng, n=300, depth_noise=0.0):
    """Points 1-8 m ahead; the goal camera `fwd` ahead, `left` to the left, turned yaw_deg left."""
    X = np.column_stack([rng.uniform(-4, 4, n), rng.uniform(-1.5, 0.132, n), rng.uniform(1.0, 8.0, n)])
    a = math.radians(yaw_deg)
    C = np.array([-left, 0.0, fwd])
    Rcg = np.stack([[math.cos(a), 0, math.sin(a)], [0, 1.0, 0], [-math.sin(a), 0, math.cos(a)]])
    Xg = (X - C) @ Rcg.T

    def px(P):
        xd, yd = distort(P[:, 0] / P[:, 2], P[:, 1] / P[:, 2])
        return np.column_stack([CAM.cx + CAM.f * xd, CAM.cy + CAM.f * yd])
    ok = (X[:, 2] > 0.3) & (Xg[:, 2] > 0.3)
    uc, ug = px(X[ok]), px(Xg[ok])
    inimg = np.all((uc > 0) & (uc < [1024, 576]), 1) & np.all((ug > 0) & (ug < [1024, 576]), 1)
    uc = uc[inimg] + rng.normal(0, 0.5, (inimg.sum(), 2))
    ug = ug[inimg] + rng.normal(0, 0.5, (inimg.sum(), 2))
    z = X[ok][inimg][:, 2] * (1 + rng.normal(0, depth_noise, inimg.sum()))
    return uc, ug, z


@pytest.mark.parametrize('fwd,left,yaw', [(2.0, 0.0, 0.0), (2.0, 0.5, 0.0), (3.0, -1.0, 20.0), (1.0, 1.0, -30.0)])
def test_bearing_heading_and_distance_from_known_geometry(fwd, left, yaw):
    rng = np.random.default_rng(1)
    m = geometry_only()
    uc, ug, z = synth(fwd, left, yaw, rng, depth_noise=0.05)
    cn, gn = m._normalize(uc, 1024), m._normalize(ug, 1024)
    e = m.solve(cn, gn)
    assert e.method == 'essential'
    assert abs(math.degrees(e.bearing) - math.degrees(math.atan2(left, fwd))) < 2.0
    assert abs(math.degrees(e.yaw) - yaw) < 1.5
    assert e.parallax_px > 50.0                       # metres from the goal: well above arrival (12 px)
    p = m.solve(cn, gn, z)                            # with 5% depth noise
    assert p.method == 'pnp'
    assert abs(p.dist - math.hypot(fwd, left)) < 0.1 * math.hypot(fwd, left) + 0.05


def test_parallax_vanishes_at_the_goal_whatever_the_heading():
    rng = np.random.default_rng(2)
    m = geometry_only()
    uc, ug, _ = synth(0.02, 0.0, 25.0, rng)          # 2 cm away, turned 25 deg
    h = m.solve(m._normalize(uc, 1024), m._normalize(ug, 1024))
    assert h.ok and h.parallax_px < 4.0
    assert abs(math.degrees(h.yaw) - 25.0) < 1.0


def test_zero_translation_is_not_mistaken_for_a_half_turn():
    """At the goal the essential matrix is undefined; a real frame against itself once read
    4097 px of parallax and a bearing of -173 deg (the 180-degree twisted pair)."""
    rng = np.random.default_rng(3)
    m = geometry_only()
    for yaw in (0.0, 12.0):
        uc, ug, _ = synth(0.0, 0.0, yaw, rng)
        h = m.solve(m._normalize(uc, 1024), m._normalize(ug, 1024))
        assert h.ok and h.parallax_px < 2.0 and abs(math.degrees(h.yaw) - yaw) < 1.0
    img = cv2.imread(os.path.join(os.path.dirname(HERE), 'config', 'goal_images', 'nyu', 'cp1.jpg'))
    photo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h = GoalMatcher(photo).match(photo)
    assert h.ok and h.parallax_px < 2.0 and abs(h.yaw) < 0.02


def texture(seed, w=1024, h=576):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 120, np.uint8)
    for _ in range(900):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(3, 25)), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_sift_path_same_view_matches_other_view_does_not():
    goal = texture(0)
    m = GoalMatcher(goal)
    same = m.match(goal)
    assert same.ok and same.inliers >= 100 and same.parallax_px < 2.0 and abs(same.yaw) < 0.02
    other = m.match(texture(1))
    assert not other.ok or other.inliers < DEFAULTS['min_inliers']


def test_sift_scales_say_behind_the_goal():
    goal = texture(0)
    small = cv2.resize(goal, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
    cur = np.full_like(goal, 120)
    y0, x0 = (goal.shape[0] - small.shape[0]) // 2, (goal.shape[1] - small.shape[1]) // 2
    cur[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    h = GoalMatcher(goal).match(cur)
    assert h.ok and abs(h.zoom - 0.9) < 0.03
    assert abs(GoalMatcher(goal).match(goal).zoom - 1.0) < 0.01


def test_a_goal_image_of_another_shape_is_flagged():
    assert GoalMatcher(texture(0, 512, 384)).warning
    assert GoalMatcher(texture(0)).warning == ''


# -- the mission's decisions, on scripted measurements -------------------------------------

def H(inliers, parallax, bearing_deg=0.0, yaw_deg=0.0, view_deg=0.0, matches=None):
    return Homing(matches=matches if matches is not None else inliers + 10, inliers=inliers,
                  bearing=math.radians(bearing_deg), yaw=math.radians(yaw_deg), parallax_px=parallax,
                  view_bearing=math.radians(view_deg), method='essential' if inliers else '')


def mission(**kw):
    return HomingMission([HomingGoal('G1', 'x', 1)], (0.0, 0.0), {'inflate_m': 0.3, 'unseen_extend_m': 0.1}, None, **kw)


def drive(m, t, x, y, yaw, h_fn, until, dt=1.0 / 3.0, t_max=200.0):
    """Tick the mission, feeding h_fn(t, yaw) as the camera; the rover turns as commanded."""
    step = None
    while t < t_max:
        m.observe_homing(t, x, y, yaw, h_fn(t, yaw))
        if m.wants_motion_check:
            m.observe_motion(t, 30.0)
        step = m.update(t, x, y, yaw)
        if until(step):
            break
        if step.command is not None:
            yaw += step.command[1] * dt
            x += step.command[0] * dt * math.cos(yaw)
            y += step.command[0] * dt * math.sin(yaw)
        t += dt
    return t, x, y, yaw, step


def test_scan_finds_the_scene_then_measures_facing_it():
    m = mission()
    # the goal's scene is only visible facing ~+90 deg (world)
    seen = lambda t, yaw: H(60, 150.0, bearing_deg=0.0) if abs(math.remainder(yaw - math.pi / 2, 2 * math.pi)) < 0.5 \
        else H(0, float('nan'), matches=3)
    t, x, y, yaw, step = drive(m, 0.0, 0.0, 0.0, 0.0, seen, lambda s: m.state == MEASURE)
    assert m.state == MEASURE
    assert abs(math.degrees(math.remainder(yaw - math.pi / 2, 2 * math.pi))) < 30.0
    assert any('scene seen' in n for n in m.notes)


def test_arrives_only_when_the_parallax_is_small():
    m = mission(align_heading=False)
    m._start_stop(0.0)
    for k in range(4):
        m.observe_homing(2.0 + 0.3 * k, 0.0, 0.0, 0.0, H(80, 40.0))
    step = m.update(3.5, 0.0, 0.0, 0.0)
    assert not step.arrived and m.state in (PULSE, ALIGN)      # near: a pulse, not a routed hop
    m2 = mission(align_heading=False)
    m2._start_stop(0.0)
    for k in range(4):
        m2.observe_homing(2.0 + 0.3 * k, 0.0, 0.0, 0.0, H(80, 4.0))
    assert m2.update(3.5, 0.0, 0.0, 0.0).arrived and m2.state == ARRIVED


def test_lined_up_but_smaller_steps_forward_instead_of_arriving():
    """A far scene: tiny parallax, but everything 8% smaller than in the photo -> behind it."""
    m = mission(align_heading=False)
    m._start_stop(0.0)
    for k in range(4):
        h = H(80, 4.0)
        h.zoom = 0.92
        m.observe_homing(2.0 + 0.3 * k, 0.0, 0.0, 0.0, h)
    step = m.update(3.5, 0.0, 0.0, 0.0)
    assert not step.arrived and m.state == PULSE and step.command[0] >= 0.0
    assert m._pulse[1] > 0.0 and any('scales 0.920' in n for n in m.notes)


def test_final_alignment_to_the_photo_heading():
    m = mission()
    m._start_stop(0.0)
    for k in range(4):
        m.observe_homing(2.0 + 0.3 * k, 0.0, 0.0, 0.0, H(80, 4.0, yaw_deg=40.0))
    step = m.update(3.5, 0.0, 0.0, 0.0)
    assert not step.arrived and m.state == ALIGN
    t, x, y, yaw, step = drive(m, 3.6, 0.0, 0.0, 0.0, lambda t, yaw: H(80, 4.0), lambda s: s.arrived)
    assert step.arrived and abs(math.degrees(yaw) - 40.0) < 15.0


def test_wheels_turning_but_view_unchanged_is_stuck():
    """After a hop, this stop's frame against the last stop's: no parallax while odometry moved."""
    m = mission(align_heading=False)
    m._start_stop(0.0)
    for k in range(4):
        m.observe_homing(2.0 + 0.3 * k, 0.0, 0.0, 0.0, H(80, 200.0))
    m.update(3.5, 0.0, 0.0, 0.0)                  # far: a routed hop
    assert m.state == 'hop'
    m._hop_end(10.0, 0.0)                          # "arrived" 1.5 m further by odometry
    assert m.wants_motion_check
    for k in range(4):
        m.observe_homing(12.0 + 0.3 * k, 1.5, 0.0, 0.0, H(80, 200.0))
    m.observe_motion(12.5, 1.0)                    # the view did not move
    step = m.update(13.5, 1.5, 0.0, 0.0)
    assert m.state == BACKOFF and step.command[0] < 0.0 and m.stucks == 1
    assert m.replanner.map.occupied()[m.replanner.map.cell(1.95, 0.0)]


def test_tilt_backs_off_and_marks_ahead():
    m = mission()
    m.block_ahead(5.0, 1.0, 0.0, 0.0, 'tilt 17 deg')
    step = m.update(5.1, 1.0, 0.0, 0.0)
    assert m.state == BACKOFF and step.command == (m.p['backoff_v'], 0.0)
    assert m.replanner.map.occupied()[m.replanner.map.cell(1.45, 0.0)]
    step = m.update(5.0 + m.p['backoff_s'] + 0.1, 0.6, 0.0, 0.0)
    assert m.state == MEASURE


def test_rejection_tightens_the_arrival():
    m = mission()
    px0 = m.p['arrive_parallax_px']
    m.confirm(1.0, False)
    assert m.p['arrive_parallax_px'] < px0 and m.state == MEASURE
    m.confirm(2.0, True)
    assert m.done


def test_one_arena_mission_end_to_end():
    import homing_sim
    r = homing_sim.run(seed=3, t_limit=400.0)
    assert r['arrived'] and r['err'] < homing_sim.JUDGE_M
