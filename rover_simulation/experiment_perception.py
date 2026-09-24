#!/usr/bin/env python3
"""The perception chain of the simple experiment, in pictures: what the camera
sees at the moment it sees the planter wall best, and each step down to the
free-space profile.

The simulator is 2-D, so this RENDERS the front camera into its world: every
pixel's ray -- from erc_perception's own calibrated camera model (f, lambda,
pitch, 13.2 cm height) -- is cast against the ground and against walls given a
height (planter 0.4 m, building 6 m). That gives a PERFECT depth image, which is
then run through free_space_node's own code (profile_from_depth: per-frame scale
fit on the ground strip, height above ground, hazard, 25 bins).

Panels
  1. bird's eye: pose, field of view, the world
  2. synthetic camera image (shading only, for orientation -- NOT a real image)
  3. depth (range along each ray): what DA3 estimates on the rover; here exact
  4. height above ground, from depth + the camera model
  5. hazard pixels (higher than the 4.5 cm ground clearance, within 3 m) and
     the ground strip the scale is fitted on
  6. the free-space profile: free_space_node's code on this depth, against
     the simplified profile controller_sim uses (obstacles.profile)
  7. homography residual, IDEAL: where each pixel moves between two frames
     0.3 m apart, minus where the ground-plane homography says it should. On
     the ground it is 0; what stands up leaves parallax. Computed from the
     geometry, not by matching image texture as tools/homography_residual.py does
  8. what is NOT here: appearance traversability (DINOv2) needs real images

    source /opt/ros/jazzy/setup.bash && source ~/lsdc_ws/install/setup.bash
    ~/lsdc/erc-omni-vla/.venv/bin/python3 experiment_perception.py [--seed 0] [--out experiment_perception.png]
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
import experiment_simple as ex  # noqa: E402

PERCEPTION = os.path.normpath(os.path.join(common.HERE, '..', '..', 'erc_perception'))
sys.path.insert(0, PERCEPTION)
from erc_perception.free_space_node import camera_rays, profile_from_depth  # noqa: E402
from erc_perception.geometry import _distort  # noqa: E402

cs, obs = common.cs, common.obs
PAR = dict(fov_deg=60.0, n_bins=25, max_range_m=3.0, clearance_m=0.045,   # config/perception.yaml
           camera_height_m=0.132, camera_pitch_deg=2.37)
SHAPE = (144, 256)                     # rows, cols of the rendered image (full frame 576x1024)
PLANTER_H, BUILDING_H = 0.40, 6.0
STEP_M = 0.3                           # second frame for the homography residual


def walls(wall):
    bx0, bx1, by0, by1 = ex.BUILDING
    b = [((bx0, by0), (bx1, by0)), ((bx1, by0), (bx1, by1)), ((bx1, by1), (bx0, by1)), ((bx0, by1), (bx0, by0))]
    return [(tuple(wall.a), tuple(wall.b), PLANTER_H, 'planter')] + [(p, q, BUILDING_H, 'building') for p, q in b]


def cast(x, y, th, d_body, h, world):
    """Range along each ray to the first surface, the hit point (world), and what was hit."""
    c, s = math.cos(th), math.sin(th)
    dx = d_body[..., 0] * c - d_body[..., 1] * s
    dy = d_body[..., 0] * s + d_body[..., 1] * c
    dz = d_body[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(dz < -1e-9, -h / dz, np.inf)            # ground
    what = np.where(np.isfinite(t), 1, 0)                     # 0 sky, 1 ground, 2 planter, 3 building
    for (ax_, ay_), (bx_, by_), H, kind in world:
        ex_, ey_ = bx_ - ax_, by_ - ay_
        den = dx * ey_ - dy * ex_
        with np.errstate(divide='ignore', invalid='ignore'):
            tt = ((ax_ - x) * ey_ - (ay_ - y) * ex_) / den       # along the ray
            uu = ((ax_ - x) * dy - (ay_ - y) * dx) / den          # along the wall
        z = h + tt * dz
        ok = (tt > 0) & (uu >= 0) & (uu <= 1) & (z >= 0) & (z <= H) & (tt < t)
        t = np.where(ok, tt, t)
        what = np.where(ok, 2 if kind == 'planter' else 3, what)
    P = np.stack([x + t * dx, y + t * dy, h + t * dz], -1)
    return t, P, what


def project(P, x, y, th, cam):
    """World points -> pixel (u, v) of a camera at rover pose (x, y, th); NaN if behind."""
    c, s = math.cos(th), math.sin(th)
    rel = P - np.array([x, y, cam.height])
    body = np.stack([rel[..., 0] * c + rel[..., 1] * s, -rel[..., 0] * s + rel[..., 1] * c, rel[..., 2]], -1)
    pc = body @ cam.rotation().T
    with np.errstate(divide='ignore', invalid='ignore'):
        xu, yu = pc[..., 0] / pc[..., 2], pc[..., 1] / pc[..., 2]
    xd, yd = _distort(xu, yu, cam.lam)
    u, v = cam.cx + cam.f * xd, cam.cy + cam.f * yd
    bad = pc[..., 2] <= 1e-6
    return np.where(bad, np.nan, u), np.where(bad, np.nan, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='experiment_perception.png')
    a = ap.parse_args()

    occ, origin = ex.global_costmap()
    route, _ = ex.global_route(occ, origin)
    wall = ex.planter_on(route)
    cs.SCENARIOS['experiment'] = cs.Scenario('experiment', (0.0, 0.0, 0.0), [list(route[1:])], [wall])
    profiles = []
    real_profile = obs.profile

    def profile(x, y, th, world, rng=None, **kw):
        out = real_profile(x, y, th, world, rng, **kw)
        profiles.append((x, y, th, *out))
        return out
    cs.obs.profile = profile
    common.run('experiment', a.seed, 'lp+ss', 1.5)
    # the moment the camera sees most of the wall, inside the mapped +-40 deg
    x, y, th, sb, sf, sc = max(profiles, key=lambda p: int(((p[4] < p[5] - 1e-3) & (np.abs(p[3]) <= 40)).sum()))

    cam, d_body, t_ground = geom = camera_rays(SHAPE, PAR['camera_pitch_deg'], PAR['camera_height_m'])
    world = walls(wall)
    rng_, P, what = cast(x, y, th, d_body, cam.height, world)
    depth = np.where(np.isfinite(rng_), rng_, 50.0) * 1.43      # relative, like DA3: the node fits the scale
    bearings, free, scale, maps = profile_from_depth(depth, geom, PAR, maps=True)

    # ideal homography residual: frame 2 is STEP_M further along the heading
    x2, y2 = x + STEP_M * math.cos(th), y + STEP_M * math.sin(th)
    u_true, v_true = project(P, x2, y2, th, cam)
    G = np.stack([x + t_ground * (d_body[..., 0] * math.cos(th) - d_body[..., 1] * math.sin(th)),
                  y + t_ground * (d_body[..., 0] * math.sin(th) + d_body[..., 1] * math.cos(th)),
                  np.zeros_like(t_ground)], -1)                  # where the pixel would be if it were ground
    u_h, v_h = project(G, x2, y2, th, cam)
    residual = np.hypot(u_true - u_h, v_true - v_h)
    residual[~np.isfinite(t_ground) | (what == 0)] = np.nan

    fig, ax = plt.subplots(2, 4, figsize=(21, 9))
    ext = [0, cam.width, cam.height_px, 0]

    # 1. bird's eye
    a0 = ax[0, 0]
    a0.add_patch(plt.Rectangle((ex.BUILDING[0], ex.BUILDING[2]), ex.BUILDING[1] - ex.BUILDING[0],
                               ex.BUILDING[3] - ex.BUILDING[2], color='0.35', label=f'building ({BUILDING_H:.0f} m)'))
    a0.plot([wall.a[0], wall.b[0]], [wall.a[1], wall.b[1]], 'g-', lw=4, label=f'planter ({PLANTER_H} m)')
    for s_ in (-60, 60):
        a0.plot([x, x + 3 * math.cos(th + math.radians(s_))], [y, y + 3 * math.sin(th + math.radians(s_))], 'b:', lw=1)
    a0.plot(x, y, 'b^', ms=11, label='rover')
    a0.plot(x2, y2, 'c^', ms=7, label=f'2nd frame (+{STEP_M} m)')
    a0.set(xlim=(x - 3, x + 5), ylim=(y - 4, y + 4))
    a0.set_aspect('equal')
    a0.grid(alpha=.3)
    a0.legend(fontsize=7, loc='lower left')
    a0.set_title('1. pose and field of view (+-60 deg, 3 m)', fontsize=10)

    # 2. synthetic image
    a1 = ax[0, 1]
    img = np.zeros(SHAPE + (3,))
    img[what == 0] = (0.75, 0.85, 0.95)
    Pf = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    chk = ((np.floor(Pf[..., 0] / 0.5) + np.floor(Pf[..., 1] / 0.5)) % 2)[..., None]
    img[what == 1] = (0.62 + 0.08 * chk[what == 1]) * np.ones(3)
    img[what == 2] = (0.25, 0.55, 0.25)
    img[what == 3] = (0.55, 0.5, 0.45)
    shade = np.clip(1.2 - 0.12 * np.nan_to_num(rng_, posinf=6), 0.5, 1)[..., None]
    a1.imshow(np.where(what[..., None] > 0, img * shade, img), extent=ext)
    a1.set_title('2. synthetic camera (shading only, NOT a real image)', fontsize=10)

    # 3. depth
    a2 = ax[0, 2]
    im = a2.imshow(np.where(what > 0, rng_, np.nan), extent=ext, cmap='viridis_r', vmin=0, vmax=6)
    fig.colorbar(im, ax=a2, fraction=.035, label='m')
    a2.set_title('3. depth (DA3 on the rover; exact here)', fontsize=10)

    # 4. height above ground
    a3 = ax[0, 3]
    hgt = np.where(what > 0, maps['height'], np.nan)
    im = a3.imshow(hgt, extent=ext, cmap='magma', vmin=-0.05, vmax=0.5)
    fig.colorbar(im, ax=a3, fraction=.035, label='m above ground')
    a3.set_title(f'4. height above ground (scale fit {scale / 1.43:.3f} of truth)', fontsize=10)

    # 5. hazard + ground strip
    a4 = ax[1, 0]
    haz = np.zeros(SHAPE + (3,))
    haz[maps['hazard']] = (0.85, 0.1, 0.1)
    haz[maps['ground_fit']] = (0.2, 0.5, 0.9)
    a4.imshow(haz, extent=ext)
    a4.set_title('5. hazard (red: > 4.5 cm, < 3 m) | scale-fit ground strip (blue)', fontsize=10)

    # 6. profile: node code vs the simulator's simplified profile
    a5 = ax[1, 1]
    w = np.diff(bearings).mean()
    a5.bar(bearings, free, width=w * 0.9, color=np.where(free < 3.0 - 1e-6, 'tab:red', 'tab:green'), alpha=.8,
           label='free_space_node code on this depth')
    a5.plot(sb, sf, 'k.-', lw=1, label='controller_sim obstacles.profile (what the sim used)')
    a5.axhline(1.2, color='tab:red', ls=':', lw=.8, label='side-step brake 1.2 m')
    a5.set_xlabel('bearing (deg, + = left)')
    a5.set_ylabel('free distance (m)')
    a5.invert_xaxis()
    a5.legend(fontsize=7)
    a5.set_title('6. FREE SPACE profile (/erc/free_space)', fontsize=10)

    # 7. homography residual
    a6 = ax[1, 2]
    im = a6.imshow(residual, extent=ext, cmap='inferno', vmin=0, vmax=np.nanpercentile(residual, 99))
    fig.colorbar(im, ax=a6, fraction=.035, label='px (full-res frame)')
    a6.set_title(f'7. homography residual, ideal ({STEP_M} m step)', fontsize=10)

    # 8. what is not here
    a7 = ax[1, 3]
    a7.axis('off')
    a7.text(0.02, 0.95,
            'In the loop on the rover today:\n'
            '  camera -> DA3 relative depth -> scale fit on ground\n'
            '  -> height above ground -> hazard -> 25-bin profile\n'
            '  -> /erc/free_space -> local costmap + side-step\n\n'
            'NOT in the loop (and not simulable here):\n'
            '  * appearance traversability (DINOv2 background\n'
            '    model, out/background.pt): catches leaf litter,\n'
            '    blind to kerbs. Needs real images.\n'
            '  * homography residual on real frames\n'
            '    (tools/homography_residual.py): a cross-check,\n'
            '    needs image texture to match.\n\n'
            'What this simulation does not have:\n'
            '  DA3 errors (scale, smoothing at edges), lens\n'
            '  vignetting, lighting, frozen frames, latency.',
            va='top', family='monospace', fontsize=8.5)

    for axx in (a1, a2, a3, a4, a6):
        axx.set_xlabel('u (px)')
        axx.set_ylabel('v (px)')
    fig.suptitle(f'Perception chain at the pose that sees the planter best: ({x:.1f}, {y:.1f}), '
                 f'heading {math.degrees(th):.0f} deg', fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=80)
    print(f'pose ({x:.2f}, {y:.2f}, {math.degrees(th):.0f} deg); scale fit {scale / 1.43:.3f} of truth; '
          f'profile node code min {free.min():.2f} m, sim profile min {np.min(sf):.2f} m')
    print('->', a.out)


if __name__ == '__main__':
    main()
