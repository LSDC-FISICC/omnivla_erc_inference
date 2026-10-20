"""Where was this photo taken, seen from here? Visual homing to a goal image.

The ERC 2026 off-road track gives each mission as a target image: drive to the place the
photo was taken from, with the camera alone (no GPS). This module answers, for one front
camera frame, where that place is relative to the rover right now. No ROS.

SIFT on both images (the fisheye frame as it is), Lowe's ratio test, then the matched
keypoints are undistorted with the camera model (erc_perception.geometry: TAREA1 f and
division-model lambda; cx/cy are the fleet prior). Two estimates, from the same inliers:

  * pose (needs metric depth for the current frame, e.g. free_space_node's UniDepthV2
    depth): the matched points become 3D in the current camera, and PnP with their goal-image
    positions gives the goal camera's pose here, in metres -- bearing, distance and the
    goal's heading relative to the rover's.
  * essential matrix (always): the same bearing and relative heading, no distance. It is
    ill-conditioned when the two cameras are close, which is what `parallax_px` is for.

`parallax_px`: the median distance, in pixels of the full frame, between each inlier's goal
position and its current position rotated into the goal camera. It is what is left of the
flow once the heading difference is taken out: zero when the rover stands where the photo was
taken, whatever it is facing, and it grows with the offset over the scene depth. It is the
arrival test. The rotation is the essential matrix's, taken from its decomposition as the one
nearest the best PURE rotation between the views (Kabsch on the inlier rays): with the rover
AT the goal there is no translation, the essential matrix is undefined, and recoverPose's
rotation came out turned 180 degrees (a real frame against itself read 4097 px). When the
pure rotation alone explains the matches within degenerate_px, that residual is the parallax.
Limit: a scene at a single depth cannot tell a sideways offset from a turn, from any camera.

`zoom`: exp(median log(size here / size in the goal image)) of the inliers' SIFT scales. ~1 at
the goal; under 1 behind it. It is a second arrival cue for far scenes, where the parallax
alone is weak: on the outdoor bags (buildings 30-100 m away) the parallax was 2-11 px from
0.5 to 5 m, and 43 of 72 pairs over 0.5 m from the goal were under 12 px; requiring
|log zoom| < 0.03 as well cut the pairs over 0.6 m that pass from ~61% to 45%, and kept
every pair within 0.3 m.

Assumption: the goal image was taken with the same camera model as the rover's (another
Mini+). A goal image of a different size is scaled to the camera's width; if its aspect
differs, `GoalMatcher.warning` says so, because then its intrinsics are unknown.
"""
import math
from dataclasses import dataclass

import numpy as np

try:
    from erc_perception.geometry import Camera, _undistort
except ImportError:                     # the same TAREA1 numbers, for a tree without erc_perception
    @dataclass(frozen=True)
    class Camera:
        f: float = 512.3
        cx: float = 512.0
        cy: float = 288.0
        lam: float = -0.4252
        width: int = 1024
        height_px: int = 576

    def _undistort(x_d, y_d, lam):
        k = 1.0 / (1.0 + lam * (x_d ** 2 + y_d ** 2))
        return x_d * k, y_d * k


DEFAULTS = dict(
    work_width=640,             # both images are matched at this width
    n_features=3000,
    ratio=0.8,                  # Lowe's ratio test
    min_inliers=15,             # fewer: no estimate at all
    ransac_px=1.5,              # essential-matrix inlier threshold, full-frame pixels
    pnp_ransac_px=8.0,          # PnP's: depth error (~5-10%) moves the reprojection that much
    max_depth_m=15.0,           # depth beyond this is not used for PnP
    min_depth_m=0.15,
    min_pnp_points=12,
    degenerate_px=3.0,          # a pure rotation explains the matches this well: the rover is at the goal
    rotation_yaw_px=20.0,       # under this parallax, the heading comes from the rotation fit
    border_frac=0.0,            # mask this fraction of rows at the bottom (rover body in view)
)


@dataclass
class Homing:
    """One frame's answer. Angles in radians in the rover's frame: 0 ahead, positive left."""
    matches: int = 0
    inliers: int = 0
    bearing: float = float('nan')       # to where the goal photo was taken
    yaw: float = float('nan')           # the goal's heading minus the rover's
    dist: float = float('nan')          # metres to it (pose estimate only)
    parallax_px: float = float('nan')
    view_bearing: float = float('nan')  # where the matched scene is (median of the inliers)
    zoom: float = float('nan')          # median SIFT scale ratio, here / goal (NaN: no scales)
    method: str = ''                    # 'pnp', 'essential' or '' (nothing usable)
    note: str = ''

    @property
    def ok(self):
        return self.method != ''


class GoalMatcher:
    def __init__(self, goal_rgb, cam=None, **params):
        import cv2
        self.cv2 = cv2
        self.cam = cam or Camera()
        self.p = dict(DEFAULTS, **params)
        self.sift = cv2.SIFT_create(nfeatures=int(self.p['n_features']))
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.warning = ''
        h, w = goal_rgb.shape[:2]
        aspect_cam = self.cam.width / self.cam.height_px
        if abs(w / h - aspect_cam) > 0.02:
            self.warning = (f'goal image is {w}x{h}, the camera {self.cam.width}x{self.cam.height_px}: '
                            'a different camera? its intrinsics are unknown, estimates will be biased')
        self.goal_w = w
        self.goal_kp, self.goal_des, self.goal_size = self._features(goal_rgb)
        self.goal_n = self._normalize(self.goal_kp, w)

    # -- features and camera ------------------------------------------------------
    def _features(self, rgb):
        cv2 = self.cv2
        h, w = rgb.shape[:2]
        s = float(self.p['work_width']) / w
        gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        if abs(s - 1.0) > 1e-6:
            gray = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        mask = None
        if self.p['border_frac'] > 0:
            mask = np.full(gray.shape, 255, np.uint8)
            mask[int(gray.shape[0] * (1 - self.p['border_frac'])):] = 0
        kp, des = self.sift.detectAndCompute(gray, mask)
        pts = np.array([k.pt for k in kp], np.float64).reshape(-1, 2) / s     # image pixels
        size = np.array([k.size for k in kp], np.float64) / s
        return pts, des, size

    def _normalize(self, pts, width):
        """Distorted pixels -> undistorted normalized coordinates (x right, y down)."""
        c = self.cam
        k = c.width / float(width)                      # to the calibrated resolution
        xd = (pts[:, 0] * k - c.cx) / c.f
        yd = (pts[:, 1] * k - c.cy) / c.f
        xu, yu = _undistort(xd, yd, c.lam)
        return np.stack([xu, yu], 1)

    def to_pinhole_px(self, pts_n, width):
        """Normalized coordinates -> pixels of the pinhole image free_space_node's depth is in
        (undistort_maps: same f and centre, scaled to `width`)."""
        c = self.cam
        s = width / float(c.width)
        return np.stack([c.cx * s + c.f * s * pts_n[:, 0], c.cy * s + c.f * s * pts_n[:, 1]], 1)

    # -- matching -----------------------------------------------------------------
    def _match(self, des):
        if des is None or self.goal_des is None or len(des) < 2 or len(self.goal_des) < 2:
            return np.zeros((0, 2), int)
        knn = self.bf.knnMatch(des, self.goal_des, k=2)
        out = []
        for m in knn:
            if len(m) == 2 and m[0].distance < self.p['ratio'] * m[1].distance:
                out.append((m[0].queryIdx, m[0].trainIdx))
        if not out:
            return np.zeros((0, 2), int)
        out = np.array(out, int)
        # one goal keypoint, one match: keep the first (knn order is by query)
        _, first = np.unique(out[:, 1], return_index=True)
        return out[np.sort(first)]

    def match(self, rgb, depth=None, depth_width=None):
        """rgb: the current front frame (any resolution with the camera's aspect).
        depth: optional metric z-depth (m) in free_space_node's pinhole geometry, at
        depth_width columns (defaults to its own width)."""
        kp, des, size = self._features(rgb)
        pairs = self._match(des)
        if len(pairs) < self.p['min_inliers']:
            return Homing(matches=len(pairs), note=f'{len(pairs)} matches')
        cur_n = self._normalize(kp, rgb.shape[1])[pairs[:, 0]]
        z = None
        if depth is not None:
            z = self.depth_at(cur_n, depth, depth_width or depth.shape[1])
        ratio = size[pairs[:, 0]] / np.maximum(self.goal_size[pairs[:, 1]], 1e-6)
        out = self.solve(cur_n, self.goal_n[pairs[:, 1]], z, ratio)
        out.matches = len(pairs)
        return out

    def depth_at(self, cur_n, depth, depth_width):
        """z-depth (m) at each normalized point, NaN off the image or on a depth edge."""
        H, W = depth.shape[:2]
        px = self.to_pinhole_px(cur_n, depth_width)
        u = np.round(px[:, 0]).astype(int)
        v = np.round(px[:, 1]).astype(int)
        z = np.full(len(u), np.nan)
        # the 3x3 neighbourhood must agree: a keypoint on a depth edge gets no depth
        for i in np.flatnonzero((u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)):
            patch = depth[v[i] - 1:v[i] + 2, u[i] - 1:u[i] + 2]
            med = float(np.median(patch))
            if np.isfinite(med) and float(np.ptp(patch)) < 0.1 * med + 0.05:
                z[i] = med
        return z

    def solve(self, cur_n, goal_n, z=None, scale_ratio=None):
        """Geometry only: matched undistorted normalized points (current, goal) and, optionally,
        the current z-depth of each (NaN where unknown) and the SIFT scale ratio of each
        (current / goal) -> Homing."""
        cv2 = self.cv2
        out = Homing(matches=len(cur_n))
        if len(cur_n) < self.p['min_inliers']:
            out.note = f'{len(cur_n)} matches'
            return out
        thr = float(self.p['ransac_px']) / self.cam.f
        E, mask = cv2.findEssentialMat(cur_n, goal_n, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                       threshold=thr)
        if E is None or mask is None:
            out.note = 'no essential matrix'
            return out
        if E.shape[0] > 3:
            E = E[:3]
        inl = mask.ravel().astype(bool)
        out.inliers = int(inl.sum())
        if inl.sum() < self.p['min_inliers']:
            out.note = f'{int(inl.sum())} inliers'
            return out
        idx = np.flatnonzero(inl)
        if scale_ratio is not None:
            out.zoom = float(np.exp(np.median(np.log(np.asarray(scale_ratio, float)[idx]))))
        a = _rays(cur_n[idx])
        b = _rays(goal_n[idx])
        Rw, res = _rotation_fit(a, b)
        par_rot = float(np.median(res) * self.cam.f)
        out.view_bearing = float(np.median(-np.arctan(cur_n[idx, 0])))   # x right -> positive left
        R, t = self._decompose(E, Rw, cur_n[idx], goal_n[idx])
        out.parallax_px = par_rot if par_rot < self.p['degenerate_px'] else \
            float(np.median(np.arccos(np.clip(np.sum((a @ R.T) * b, axis=1), -1.0, 1.0))) * self.cam.f)
        C = -R.T @ t                                    # goal camera centre, current camera frame
        use = Rw if out.parallax_px < self.p['rotation_yaw_px'] else R
        axis = use.T @ np.array([0.0, 0.0, 1.0])        # goal optical axis, current camera frame
        out.bearing = float(math.atan2(-C[0], C[2]))
        out.yaw = float(math.atan2(-axis[0], axis[2]))
        out.method = 'essential'

        if z is not None:
            pose = self._pnp(cur_n[idx], goal_n[idx], np.asarray(z, float)[idx])
            if pose is not None:
                R2, C2, n_pnp = pose
                axis2 = R2.T @ np.array([0.0, 0.0, 1.0])
                out.bearing = float(math.atan2(-C2[0], C2[2]))
                out.yaw = float(math.atan2(-axis2[0], axis2[2]))
                out.dist = float(math.hypot(C2[0], C2[2]))
                out.method = 'pnp'
                out.note = f'pnp {n_pnp}/{len(idx)}'
        return out

    def _decompose(self, E, Rw, cur_n, goal_n):
        """The essential matrix's (R, t) nearest the pure-rotation fit Rw, t's sign by cheirality."""
        cv2 = self.cv2
        R1, R2, t = cv2.decomposeEssentialMat(E)
        R = min((R1, R2), key=lambda Rk: _angle(Rk @ Rw.T))
        t = t.ravel()
        best, best_n = t, -1
        for tt in (t, -t):
            P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = np.hstack([R, tt.reshape(3, 1)])
            X = cv2.triangulatePoints(P1, P2, cur_n.T.astype(np.float64), goal_n.T.astype(np.float64))
            X = X[:3] / np.where(np.abs(X[3]) < 1e-12, 1e-12, X[3])
            z1 = X[2]
            z2 = (R @ X + tt.reshape(3, 1))[2]
            n = int(((z1 > 0) & (z2 > 0)).sum())
            if n > best_n:
                best, best_n = tt, n
        return R, best

    def _pnp(self, cur_n, goal_n, z):
        cv2 = self.cv2
        ok = np.isfinite(z) & (z > self.p['min_depth_m']) & (z < self.p['max_depth_m'])
        if ok.sum() < self.p['min_pnp_points']:
            return None
        X = np.column_stack([cur_n[ok] * z[ok, None], z[ok]])
        good, rvec, tvec, inl = cv2.solvePnPRansac(
            X.astype(np.float64), goal_n[ok].astype(np.float64), np.eye(3), None,
            reprojectionError=float(self.p['pnp_ransac_px']) / self.cam.f, iterationsCount=200,
            flags=cv2.SOLVEPNP_EPNP)
        if not good or inl is None or len(inl) < self.p['min_pnp_points']:
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(X[inl], goal_n[ok][inl].astype(np.float64), np.eye(3), None,
                                          rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        return R, -R.T @ tvec.ravel(), len(inl)


def _rays(pts_n):
    r = np.column_stack([pts_n, np.ones(len(pts_n))])
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _angle(R):
    return math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))


def _rotation_fit(a, b, iters=3):
    """R minimising |R a - b| over unit rays (Kabsch), refitted without its worst residuals.
    -> (R, angular residual of every ray, rad)."""
    keep = np.ones(len(a), bool)
    R = np.eye(3)
    for _ in range(iters):
        U, _, Vt = np.linalg.svd(a[keep].T @ b[keep])
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        res = np.arccos(np.clip(np.sum((a @ R.T) * b, axis=1), -1.0, 1.0))
        keep = res <= max(3.0 * float(np.median(res)), 1e-4)
    return R, res


def photo_scores(matchers, rgb, ransac_px=2.0):
    """How well one frame matches each goal photo, taken with ANY camera: SIFT inliers of a
    fundamental matrix (pixels, no intrinsics), and where in the frame they are (median bearing,
    degrees, left-positive, through the rover's camera model): one (inliers, bearing) per
    matcher. The frame's features are computed once."""
    if not matchers:
        return []
    cv2 = matchers[0].cv2
    kp, des, _ = matchers[0]._features(rgb)
    out = []
    cur_n = matchers[0]._normalize(kp, rgb.shape[1]) if len(kp) else np.zeros((0, 2))
    for m in matchers:
        pairs = m._match(des)
        if len(pairs) < 8:
            out.append((0, float('nan')))
            continue
        ww = float(m.p['work_width'])     # both images at the width they were matched at
        a = kp[pairs[:, 0]] * (ww / rgb.shape[1])
        b = m.goal_kp[pairs[:, 1]] * (ww / m.goal_w)
        F, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, ransac_px, 0.999)
        if F is None or mask is None or not mask.any():
            out.append((0, float('nan')))
            continue
        inl = mask.ravel().astype(bool)
        brg = float(np.degrees(np.median(-np.arctan(cur_n[pairs[inl, 0], 0]))))
        out.append((int(inl.sum()), brg))
    return out
