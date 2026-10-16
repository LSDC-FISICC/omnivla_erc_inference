"""People are not part of the building: keep them off a LocalCostmap built from the free-space profile.

YOLO26 sees a person (object_detector_node, label 'person': the bearings of the box's bottom corners
and a range); the depth profile sees the same person as an obstacle. On 12-oct two people walking up
the corridor went into the map as a wall across it, and it stayed there after they had gone.

  observe  a frame's people, seen from (x, y, yaw) at time t. What the map already holds where each
           one stands -- profiles taken before YOLO's frame arrived, or as they walked out of the
           masked bearings -- loses its hit evidence (log-odds down to 0, not to free: the floor
           there is cleared by the next profile that sees it, as the rest of the map is).
  filter   a profile taken at time t with heading yaw: the bins within a person's bearings (plus a
           margin: they walk, and YOLO's frame is not the depth model's) of a sighting within mask_s
           report no hit, and free ground only up to clear_m short of the person.

What the rover drives by is the profile as it came: this only decides what is REMEMBERED.
No ROS: explore_search (the mission) and local_map_node (the view) use it.
"""
import math

import numpy as np

DEFAULTS = dict(
    mask_s=0.8,       # a profile within this of a sighting has that person's bins kept off the map
    mask_deg=4.0,     # margin each side of the box ...
    mask_m=0.4,       # ... plus this much sideways at their range
    clear_m=0.6,      # the hits within this of a sighted person, along their box, are dropped
)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PeopleMask:
    def __init__(self, **kw):
        self.p = dict(DEFAULTS)
        self.p.update({k: v for k, v in kw.items() if k in DEFAULTS})
        self.sightings = []          # (t, yaw, right_deg, left_deg, range_m or None)

    def span(self, q, yaw):
        """A sighting's bearing span (deg) in the frame of a profile taken at heading yaw, with margins."""
        _t, yaw_q, right, left, rng = q
        off = math.degrees(_wrap(yaw_q - yaw))
        m = self.p['mask_deg'] + math.degrees(math.atan2(self.p['mask_m'], max(rng or 1.0, 0.5)))
        return right + off - m, left + off + m

    def observe(self, t, x, y, yaw, people, cmap=None):
        """people: [(right_deg, left_deg, range_m or None), ...], bearings left-positive."""
        self.sightings = [q for q in self.sightings if abs(t - q[0]) <= self.p['mask_s']]
        for right, left, rng in people:
            right, left = min(right, left), max(right, left)
            self.sightings.append((t, yaw, right, left, rng))
            if rng is not None and cmap is not None:
                self._clear(cmap, x, y, yaw, right, left, rng)

    def filter(self, t, yaw, b, r, h):
        """-> (ranges, hit) of a profile with the people taken out."""
        near = [q for q in self.sightings if abs(t - q[0]) <= self.p['mask_s']]
        if not near:
            return r, h
        r, h = np.array(r, float), np.array(h, bool)
        b = np.asarray(b, float)
        for q in near:
            lo, hi = self.span(q, yaw)
            k = (b >= lo) & (b <= hi)
            h[k] = False
            r[k] = np.minimum(r[k], max(0.0, (q[4] or 0.0) - self.p['clear_m']))
        return r, h

    def _clear(self, m, x, y, yaw, right, left, rng):
        res = m.p['resolution_m']
        lo, hi = self.span((0.0, yaw, right, left, rng), yaw)
        r0, r1 = max(0.0, rng - self.p['clear_m']), rng + self.p['clear_m']
        k = int(math.ceil(r1 / res)) + 1
        rc, cc = (int(v) for v in m.cell(x, y))
        rows = np.arange(max(0, rc - k), min(m.h, rc + k + 1))
        cols = np.arange(max(0, cc - k), min(m.w, cc + k + 1))
        if len(rows) == 0 or len(cols) == 0:
            return
        gx = m.ox + (cols[None, :] + 0.5) * res - x
        gy = m.oy + (rows[:, None] + 0.5) * res - y
        d = np.hypot(gx, gy)
        brg = np.degrees(_wrap(np.arctan2(gy, gx) - yaw))
        sel = (d >= r0) & (d <= r1) & (brg >= lo) & (brg <= hi)
        if sel.any():
            win = m.L[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
            win[sel] = np.minimum(win[sel], 0.0)
            m._dist = None
