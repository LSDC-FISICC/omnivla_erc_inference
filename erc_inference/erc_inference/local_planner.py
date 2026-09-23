"""Local obstacle map from /erc/free_space, and local replanning onto the global route.

Why. The first field runs of obstacle_sidestep (mission_carrot_sidestep*,
2026-09-22) braked 23 times, every one of the nine checked on camera a real
obstacle -- planter walls and stairs OSM does not have. But they came in runs:
five times against the same hedge, four against the same stairs, 2-2.5 m apart.
The profile has no memory and the carrot keeps pointing at the global route, so
after each side-step the rover was pulled straight back into the wall: 90 m
driven for 15 m of progress in 6 minutes. controller_sim's chicane shows the
same saw-tooth, and ends in a collision at the wall's end.

What this does:
  LocalCostmap   log-odds grid in the leg's own local frame (the east/north
                 metres checkpoint_controller_node already drives the carrot
                 in). Each profile adds hit evidence at the reported range and
                 free evidence along the ray before it. Bins that report
                 nothing (range at the sensor's cap) add free evidence only.
  LocalReplanner when the route ahead crosses an inflated obstacle, plans A*
                 from the rover to the first point of the GLOBAL route past
                 the blockage -- the waypoint it could not reach -- and splices
                 that path in front of the rest of the route. The carrot then
                 follows the spliced route exactly as it follows any route.

What it does not do: it never commands the rover. Side-step/stop in
carrot_controller_node stay as the reflex for what the map has not caught.

Known limits, open:
  * The map is only as good as position and heading. The compass lags ~0.7-1.2 s
    (Mission17sept 4, and the 2.7 s turn onset of 22-sept); during a turn a
    lagged heading smears every hit sideways, so scans are NOT integrated while
    the heading moves faster than max_turn_rate_dps. A GPS jump moves the rover
    relative to everything already mapped; nothing here detects that.
  * The profile's range comes from DA3 with a per-frame scale fit. Its
    distances have never been checked against a tape in the field.

No ROS, no torch: checkpoint_controller_node and controller_sim import it.
"""
import heapq
import math

import numpy as np
from scipy import ndimage

DEFAULTS = dict(
    resolution_m=0.2,
    margin_m=12.0,            # map extends this far around the route's bounding box
    hit_logodds=0.9,
    free_logodds=-0.4,
    min_logodds=-2.0,
    max_logodds=3.5,
    occupied_logodds=1.5,     # two consistent hits
    free_stop_before_hit_m=0.3,   # do not clear the cells right in front of a hit
    max_free_m=2.5,           # beyond this the profile is not trusted to call ground free
    max_turn_rate_dps=12.0,   # skip scans while the compass turns faster
    map_fov_deg=40.0,         # only bins within this bearing enter the map: the
                              # periphery reads ~1.7 m with nothing there (TAREA5 6c.4)
    inflate_m=0.45,           # lethal: robot half-length + margin (footprint 0.25 m wide).
    soft_m=1.2,               # extra cost fades out to this distance. 0.35/1.0 planned
                              # paths 0.5 m past wall ends -- inside the side-step's
                              # +-20 deg x 1.2 m brake sector, so the reflex fought the
                              # plan (controller_sim chicane: 5 side-steps, give-up)
    soft_weight=4.0,
    unseen_weight=2.0,        # extra cost of a cell the camera has never looked at:
                              # a path around a planter must not assume its unseen
                              # sides are open (controller_sim 'planter')
    unseen_extend_m=0.8,      # unseen cells this much beyond inflate_m of an obstacle
                              # are lethal: a wall seen in part probably continues
                              # where the camera has not looked (the +-40 deg map
                              # FOV misses a wall's end as the rover turns off it)
    check_ahead_m=5.0,        # route length ahead checked for obstacles
    check_period_s=0.5,
    min_replan_interval_s=1.0,
    search_beyond_m=30.0,     # how far along the route to look for a free rejoin point
    rejoin_beyond_m=3.0,      # target this far past the end of the blockage: 1.0
                              # put it inside a planter whose back was not yet seen
    window_margin_m=8.0,      # A* window around rover + target
    ignore_within_m=0.6,      # a block nearer than this is the reflex's business
)


class LocalCostmap:
    def __init__(self, xmin, ymin, xmax, ymax, **kw):
        self.p = dict(DEFAULTS)
        self.p.update(kw)
        r = self.p['resolution_m']
        m = self.p['margin_m']
        self.ox, self.oy = xmin - m, ymin - m
        self.w = int(math.ceil((xmax - xmin + 2 * m) / r))
        self.h = int(math.ceil((ymax - ymin + 2 * m) / r))
        self.L = np.zeros((self.h, self.w), np.float32)
        self.seen = np.zeros((self.h, self.w), bool)
        self._dist = None      # distance to the nearest occupied cell, m (cached)

    # -- geometry ----------------------------------------------------------
    def cell(self, x, y):
        r = self.p['resolution_m']
        return (np.floor((np.asarray(y) - self.oy) / r).astype(int),
                np.floor((np.asarray(x) - self.ox) / r).astype(int))

    def centre(self, row, col):
        r = self.p['resolution_m']
        return self.ox + (col + 0.5) * r, self.oy + (row + 0.5) * r

    def _inside(self, rows, cols):
        return (rows >= 0) & (rows < self.h) & (cols >= 0) & (cols < self.w)

    # -- evidence ----------------------------------------------------------
    def update(self, x, y, yaw, bearings_deg, ranges, hit):
        """One profile seen from (x, y) with ENU yaw (rad, CCW from east).

        bearings_deg left-positive, ranges in m, hit[i] True where the bin
        reports an obstacle (False: nothing within the sensor's reach).
        """
        p = self.p
        res = p['resolution_m']
        b = np.asarray(bearings_deg, float)
        k = np.abs(b) <= p['map_fov_deg']
        ang = yaw + np.radians(b[k])
        rng = np.asarray(ranges, float)[k]
        hit = np.asarray(hit, bool)[k] & np.isfinite(rng)
        free_to = np.where(hit, rng - p['free_stop_before_hit_m'], rng)
        free_to = np.clip(free_to, 0.0, p['max_free_m'])
        # free evidence: every cell a ray crosses, once per scan
        steps = np.arange(0.0, p['max_free_m'] + 1e-9, res * 0.5)
        fx = x + np.cos(ang)[:, None] * steps[None, :]
        fy = y + np.sin(ang)[:, None] * steps[None, :]
        keep = steps[None, :] <= free_to[:, None]
        rows, cols = self.cell(fx[keep], fy[keep])
        ok = self._inside(rows, cols)
        idx = np.unique(rows[ok] * self.w + cols[ok])
        self.L.flat[idx] += p['free_logodds']
        self.seen.flat[idx] = True
        # hit evidence
        hx = x + np.cos(ang[hit]) * rng[hit]
        hy = y + np.sin(ang[hit]) * rng[hit]
        rows, cols = self.cell(hx, hy)
        ok = self._inside(rows, cols)
        idx = np.unique(rows[ok] * self.w + cols[ok])
        self.L.flat[idx] += p['hit_logodds']
        self.seen.flat[idx] = True
        np.clip(self.L, p['min_logodds'], p['max_logodds'], out=self.L)
        self._dist = None

    def occupied(self):
        return self.L >= self.p['occupied_logodds']

    def distance(self):
        """Metres from every cell to the nearest occupied one (inf if none)."""
        if self._dist is None:
            occ = self.occupied()
            if not occ.any():
                self._dist = np.full(occ.shape, np.inf, np.float32)
            else:
                self._dist = (ndimage.distance_transform_edt(~occ)
                              * self.p['resolution_m']).astype(np.float32)
        return self._dist

    def clearance_at(self, x, y):
        rows, cols = self.cell(x, y)
        rows, cols = np.atleast_1d(rows), np.atleast_1d(cols)
        out = np.full(rows.shape, np.inf, np.float32)
        ok = self._inside(rows, cols)
        out[ok] = self.distance()[rows[ok], cols[ok]]
        return out

    # -- planning ----------------------------------------------------------
    def plan(self, start, goal, extra_lethal=None):
        """A* from start to goal (x, y). -> list of (x, y) or None.

        8-connected, lethal within inflate_m, cost rising towards obstacles out
        to soft_m. Unknown cells cost as free. The start is cleared of
        inflation (the rover may already stand inside it; it has to get out).
        extra_lethal(xs, ys) -> bool array marks cells lethal for other
        reasons (the static OSM costmap).
        """
        p = self.p
        res = p['resolution_m']
        wm = p['window_margin_m']
        x0, x1 = min(start[0], goal[0]) - wm, max(start[0], goal[0]) + wm
        y0, y1 = min(start[1], goal[1]) - wm, max(start[1], goal[1]) + wm
        (r0, r1), (c0, c1) = self.cell([x0, x1], [y0, y1])
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, self.h - 1), min(c1, self.w - 1)
        if r1 <= r0 or c1 <= c0:
            return None
        dist = self.distance()[r0:r1 + 1, c0:c1 + 1]
        unseen = ~self.seen[r0:r1 + 1, c0:c1 + 1]
        lethal = (dist <= p['inflate_m']) | (unseen & (dist <= p['inflate_m'] + p['unseen_extend_m']))
        cost = 1.0 + p['soft_weight'] * np.clip(
            (p['soft_m'] - dist) / max(p['soft_m'] - p['inflate_m'], 1e-6), 0.0, 1.0)
        cost = cost + p['unseen_weight'] * unseen
        if extra_lethal is not None:
            rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
            xs, ys = self.centre(rr, cc)
            lethal |= np.asarray(extra_lethal(xs, ys), bool)
        H, W = lethal.shape
        (sr, sc), (gr, gc) = self.cell(*start), self.cell(*goal)
        sr, sc, gr, gc = int(sr) - r0, int(sc) - c0, int(gr) - r0, int(gc) - c0
        if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
            return None
        # let the rover out of the inflation it stands in, never through a hit
        k = int(math.ceil(p['inflate_m'] / res)) + 1
        rr, cc = np.ogrid[-sr:H - sr, -sc:W - sc]
        near_start = rr * rr + cc * cc <= k * k
        lethal = lethal & ~(near_start & (dist > res))
        if lethal[gr, gc]:
            return None

        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                 (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
        g = np.full((H, W), np.inf)
        parent = np.full((H, W), -1, np.int64)
        g[sr, sc] = 0.0
        heap = [(math.hypot(gr - sr, gc - sc), sr, sc)]
        closed = np.zeros((H, W), bool)
        while heap:
            _f, r, c = heapq.heappop(heap)
            if closed[r, c]:
                continue
            if (r, c) == (gr, gc):
                break
            closed[r, c] = True
            for dr, dc, step in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W) or lethal[nr, nc] or closed[nr, nc]:
                    continue
                ng = g[r, c] + step * 0.5 * (cost[r, c] + cost[nr, nc])
                if ng < g[nr, nc]:
                    g[nr, nc] = ng
                    parent[nr, nc] = r * W + c
                    heapq.heappush(heap, (ng + math.hypot(gr - nr, gc - nc), nr, nc))
        if not np.isfinite(g[gr, gc]):
            return None
        cells = [(gr, gc)]
        while cells[-1] != (sr, sc):
            i = parent[cells[-1]]
            cells.append((int(i // W), int(i % W)))
        cells.reverse()
        path = [self.centre(r + r0, c + c0) for r, c in cells]
        path[0], path[-1] = tuple(start), tuple(goal)
        return self._shortcut(path)

    def _clear_segment(self, a, b, need):
        n = max(2, int(math.ceil(math.dist(a, b) / (0.4 * self.p['resolution_m']))))
        t = np.linspace(0.0, 1.0, n)
        xs, ys = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        clear = self.clearance_at(xs, ys)
        rows, cols = self.cell(xs, ys)
        ok = self._inside(rows, cols)
        seen = np.zeros(len(xs), bool)
        seen[ok] = self.seen[rows[ok], cols[ok]]
        # the same rule A* followed: unseen ground next to a wall is not free
        far = clear > self.p['inflate_m'] + self.p['unseen_extend_m']
        return bool(np.all((clear > need) & (seen | far)))

    def _shortcut(self, path):
        """Fewest points such that each straight segment keeps soft_m clearance
        where the A* path itself did, and at least inflate_m everywhere."""
        if len(path) <= 2:
            return path
        clear = self.clearance_at(np.array([q[0] for q in path]), np.array([q[1] for q in path]))
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j_best = i + 1
            for j in range(i + 2, len(path)):
                need = min(self.p['soft_m'], float(clear[i:j + 1].min())) - 1e-3
                need = max(need, self.p['inflate_m'])
                if i == 0 and float(clear[0]) <= need:
                    need = self.p['inflate_m'] * 0.5   # leaving the start's inflation
                if not self._clear_segment(path[i], path[j], need):
                    break
                j_best = j
            out.append(path[j_best])
            i = j_best
        return out


def route_points(pts, cum, s0, s1, step):
    """Route positions (N, 2) and their arc lengths from s0 to s1."""
    s = np.arange(s0, min(s1, float(cum[-1])) + 1e-9, step)
    if len(s) == 0:
        s = np.array([min(s0, float(cum[-1]))])
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    return np.stack([x, y], axis=1), s


class LocalReplanner:
    """The policy around LocalCostmap: when to integrate, when and where to replan."""

    def __init__(self, pts, **kw):
        self.p = dict(DEFAULTS)
        self.p.update(kw)
        pts = np.asarray(pts, float)
        self.map = LocalCostmap(pts[:, 0].min(), pts[:, 1].min(),
                                pts[:, 0].max(), pts[:, 1].max(), **self.p)
        self._yaws = []            # (t, yaw) for the turn-rate gate
        self._last_check = -math.inf
        self._last_plan = -math.inf
        self.replans = 0
        self.skipped_scans = 0
        self.scans = 0

    def observe(self, t, x, y, yaw, bearings_deg, ranges, hit):
        """Integrate one profile unless the heading is moving too fast to trust."""
        self._yaws = [(tt, a) for tt, a in self._yaws if t - tt <= 1.0] + [(t, yaw)]
        if len(self._yaws) >= 2 and self._yaws[-1][0] - self._yaws[0][0] > 0.3:
            (ta, a), (tb, b) = self._yaws[0], self._yaws[-1]
            d = (b - a + math.pi) % (2 * math.pi) - math.pi
            if abs(math.degrees(d)) / (tb - ta) > self.p['max_turn_rate_dps']:
                self.skipped_scans += 1
                return False
        self.map.update(x, y, yaw, bearings_deg, ranges, hit)
        self.scans += 1
        return True

    def check(self, t, x, y, pts, cum, s_proj, extra_lethal=None):
        """-> (new_pts or None, note). new_pts starts at (x, y); arc length restarts at 0."""
        p = self.p
        if t - self._last_check < p['check_period_s']:
            return None, ''
        self._last_check = t
        pts = np.asarray(pts, float)
        ahead, s = route_points(pts, cum, s_proj, s_proj + p['check_ahead_m'], 0.1)
        blocked = self.map.clearance_at(ahead[:, 0], ahead[:, 1]) <= p['inflate_m']
        if not blocked.any():
            return None, ''
        first = int(np.argmax(blocked))
        if math.hypot(ahead[first, 0] - x, ahead[first, 1] - y) < p['ignore_within_m']:
            return None, ''
        if t - self._last_plan < p['min_replan_interval_s']:
            return None, ''
        self._last_plan = t
        # the first route point past the blockage that is itself clear
        far, s_far = route_points(pts, cum, s[first], s_proj + p['search_beyond_m'], 0.1)
        clear = self.map.clearance_at(far[:, 0], far[:, 1]) > p['soft_m']
        k = None
        for i in range(len(clear)):
            if not clear[i]:
                continue
            j = int(np.searchsorted(s_far, s_far[i] + p['rejoin_beyond_m']))
            if j < len(clear) and clear[i:j + 1].all():
                k = j
                break
        if k is None:
            k = len(s_far) - 1          # route end within the search: aim there
        target, s_t = far[k], float(s_far[k])
        path = self.map.plan((x, y), tuple(target), extra_lethal)
        if path is None:
            return None, (f'[local-plan] blocked {s[first] - s_proj:.1f} m ahead; '
                          f'no path to route s={s_t:.1f} m')
        rest = pts[cum > s_t + 1e-6]
        new = np.vstack([np.asarray(path, float), rest]) if len(rest) else np.asarray(path, float)
        # drop zero-length steps (path ends exactly on a route vertex)
        keep = np.concatenate([[True], np.hypot(*np.diff(new, axis=0).T) > 1e-3])
        new = new[keep]
        self.replans += 1
        return new, (f'[local-plan] blocked {s[first] - s_proj:.1f} m ahead; rejoin route '
                     f'{s_t - s_proj:.1f} m ahead via {len(path)} points '
                     f'({sum(math.dist(a, b) for a, b in zip(path, path[1:])):.1f} m)')
