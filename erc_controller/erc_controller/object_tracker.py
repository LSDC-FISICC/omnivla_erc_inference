"""Objects on the map, each ONE track: which detection is which object, and where it stands.

For the search missions (explore_search, local_map_node), fed by object_detector_node. It
replaces loop_patrol.ConeMap there. That one joins a detection to the nearest cluster within
1.5 m, which is right for cones of different colours seen a few metres off, and wrong for
several objects of one class ranged from an image: the range of a detection is good to a
fraction of itself, not to a fixed 1.5 m, so one potted plant 8 m ahead became a new cluster
every time its range moved (2026-10-09 on the rover: eight markers for one plant).

  observe()   one camera frame's detections at once. Each is matched to the track it is
              consistent with -- along the line of sight it was seen on (within bearing_gate_deg)
              and in range (within gate_sigmas of the two uncertainties) -- best matches
              first, one detection per track and one track per detection: two objects in the
              same frame are never the same object. What matches nothing starts a track.
  position    the mean of a track's observations weighted by 1/sigma^2, with sigma growing
              with the range it was seen from (range_sigma_rel): the close looks decide, the
              far ones only say roughly where.
  confirmed   after confirm_hits observations; a track that never gets there is dropped
              unconfirmed_ttl_s after its last sighting (a detection that came and went).
  merge       two tracks that end up within merge_m of each other, and were never seen in the
              same frame, are one object seen twice: merged.

No ROS. Pose and detections in whatever frame the caller maps in.
"""
import collections
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

DEFAULTS = dict(
    range_sigma_rel=0.12,       # 1 sigma of a detection's position, as a fraction of its range: the
                                # depth model read 4.4 m for a pot at ~4.9 m (2026-10-09)
    range_sigma_min=0.2,        # m, however close
    bearing_gate_deg=8.0,       # a track this far off the detection's line of sight is another object
    bearing_gate_m=0.6,         # ... or this far sideways, whichever allows more (close up)
    gate_sigmas=3.0,
    gate_floor_m=0.4,
    confirm_hits=3,
    unconfirmed_ttl_s=8.0,
    merge_m=0.7,
    max_observations=300,
)


@dataclass
class Track:
    id: int
    cls: str
    obs: collections.deque = field(default_factory=collections.deque)    # (x, y, sigma, t, odometer)
    hits: int = 0
    confirmed: bool = False
    visited: bool = False
    name: str = ''                # set by the mission once shown ('cone3') or given up ('rejected')
    last_t: float = -math.inf
    last_frame: int = -1
    seen_with: set = field(default_factory=set)      # ids of tracks seen in a same frame: never merged with

    @property
    def points(self):             # loop_patrol.ConeCluster's name for the observations
        return self.obs

    def _mean(self, obs):
        a = np.array([(o[0], o[1], o[2]) for o in obs], float)
        w = 1.0 / a[:, 2] ** 2
        return (a[:, :2] * w[:, None]).sum(0) / w.sum()

    def estimate(self, n=None) -> np.ndarray:
        return self._mean(self.obs)

    def sigma(self) -> float:
        return 1.0 / math.sqrt(sum(1.0 / o[2] ** 2 for o in self.obs))

    def live(self, t, window) -> Optional[np.ndarray]:
        """The position from the last `window` s alone (the approach follows what is in view), or
        None with fewer than 2 observations there."""
        if t is None:
            return None
        recent = [o for o in self.obs if t - o[3] <= window]
        return self._mean(recent) if len(recent) >= 2 else None


class ObjectTracker:
    def __init__(self, **kw):
        self.p = dict(DEFAULTS)
        self.p.update({k: v for k, v in kw.items() if k in DEFAULTS})
        self.clusters: List[Track] = []          # loop_patrol.ConeMap's name for them
        self._next_id = 1
        self._frame = 0

    def _sigma(self, rng):
        return max(self.p['range_sigma_min'], self.p['range_sigma_rel'] * rng)

    def _consistent(self, track, x, y, bearing, rng, sig):
        """How far the track is from where this detection puts its object, in sigmas; inf if the
        track is off the detection's line of sight or outside the range gate."""
        e = track.estimate()
        dx, dy = e[0] - x, e[1] - y
        dist = math.hypot(dx, dy)
        off = abs((math.atan2(dy, dx) - bearing + math.pi) % (2 * math.pi) - math.pi)
        if dist > 0.3 and off > max(math.radians(self.p['bearing_gate_deg']),
                                    math.atan2(self.p['bearing_gate_m'], dist)):
            return math.inf
        s = math.hypot(sig, max(track.sigma(), 0.15))
        d = math.hypot(x + rng * math.cos(bearing) - e[0], y + rng * math.sin(bearing) - e[1])
        return d / s if d <= self.p['gate_sigmas'] * s + self.p['gate_floor_m'] else math.inf

    def observe(self, x, y, yaw, detections, t, odometer=0.0) -> List[Track]:
        """detections: [(class, bearing_deg left-positive, range_m), ...] of ONE frame taken from
        pose (x, y, yaw). -> the track each detection went to, in the same order."""
        self._frame += 1
        p = self.p
        dets = [(cls, yaw + math.radians(b), float(r), self._sigma(float(r))) for cls, b, r in detections]
        pairs = []
        for i, (cls, brg, rng, sig) in enumerate(dets):
            for tr in self.clusters:
                if tr.cls == cls and tr.name != 'rejected':
                    d = self._consistent(tr, x, y, brg, rng, sig)
                    if math.isfinite(d):
                        pairs.append((d, i, tr))
        out: List[Optional[Track]] = [None] * len(dets)
        taken = set()
        for d, i, tr in sorted(pairs, key=lambda q: q[0]):
            if out[i] is None and tr.id not in taken:
                out[i] = tr
                taken.add(tr.id)
        for i, (cls, brg, rng, sig) in enumerate(dets):
            tr = out[i]
            if tr is None:
                tr = out[i] = Track(self._next_id, cls)
                self._next_id += 1
                self.clusters.append(tr)
            tr.obs.append((x + rng * math.cos(brg), y + rng * math.sin(brg), sig, t, odometer))
            while len(tr.obs) > p['max_observations']:
                tr.obs.popleft()
            tr.hits += 1
            tr.confirmed = tr.confirmed or tr.hits >= p['confirm_hits']
            tr.last_t, tr.last_frame = t, self._frame
        ids = {tr.id for tr in out}
        for tr in out:
            tr.seen_with |= ids - {tr.id}
        self._prune(t)
        self._merge(out)
        return out

    def _prune(self, t):
        self.clusters = [tr for tr in self.clusters
                         if tr.confirmed or tr.visited or t - tr.last_t <= self.p['unconfirmed_ttl_s']]

    def _merge(self, updated):
        for a in updated:
            if a not in self.clusters:
                continue
            for b in list(self.clusters):
                if b is a or b.cls != a.cls or b.id in a.seen_with or a.id in b.seen_with \
                        or 'rejected' in (a.name, b.name):
                    continue
                if float(np.hypot(*(a.estimate() - b.estimate()))) > self.p['merge_m']:
                    continue
                keep, gone = (b, a) if (b.visited and not a.visited) or (b.visited == a.visited and b.id < a.id) else (a, b)
                keep.obs.extend(gone.obs)
                keep.hits += gone.hits
                keep.confirmed = keep.confirmed or gone.confirmed or keep.hits >= self.p['confirm_hits']
                keep.seen_with |= gone.seen_with
                keep.last_t = max(keep.last_t, gone.last_t)
                self.clusters.remove(gone)
                if gone is a:
                    break

    def forget(self, before_odometer):
        """Drop the tracks not shown yet whose last sighting was before the rover had travelled
        that far: the map around them has been forgotten too, and the pose has drifted since."""
        self.clusters = [tr for tr in self.clusters if tr.visited or (tr.obs and tr.obs[-1][4] >= before_odometer)]

    def add_known(self, cls, xy, name='', visited=False, sigma=0.3) -> Track:
        """A track from memory (a resumed mission): already confirmed."""
        tr = Track(self._next_id, cls, hits=self.p['confirm_hits'], confirmed=True, visited=visited, name=name)
        self._next_id += 1
        tr.obs.append((float(xy[0]), float(xy[1]), sigma, -math.inf, -math.inf))
        self.clusters.append(tr)
        return tr


def _selftest():
    rng = np.random.default_rng(1)
    plants = np.array([[9.0, 0.6], [9.6, -0.7], [3.0, 4.0]])          # two of them 1.4 m apart
    tk = ObjectTracker()
    x = 0.0
    for k in range(120):                                             # driving up to them, 4 Hz
        x = min(7.5, 0.0625 * k)
        dets = []
        for px, py in plants:
            r = math.hypot(px - x, py)
            b = math.degrees(math.atan2(py, px - x))
            if abs(b) < 50 and r < 8.0 and rng.random() < 0.8:
                dets.append(('potted plant', b + rng.normal(0, 1.5), r * (0.92 + rng.normal(0, 0.08))))
        if rng.random() < 0.05:                                      # a detection that comes and goes
            dets.append(('potted plant', rng.uniform(-40, 40), rng.uniform(2, 7)))
        tk.observe(x, 0.0, 0.0, dets, 0.25 * k, x)
    conf = [t for t in tk.clusters if t.confirmed]
    est = np.array([t.estimate() for t in conf])
    near = [float(np.hypot(*(est - pl).T).min()) for pl in plants[:2]]
    assert len(conf) == 2, [(t.id, t.hits, t.estimate().round(1)) for t in tk.clusters]
    assert max(near) < 0.5, near
    print(f'object_tracker selftest ok: {len(conf)} tracks for the 2 plants in view, {near[0]:.2f} and {near[1]:.2f} m off; '
          f'{len(tk.clusters) - len(conf)} unconfirmed left')


if __name__ == '__main__':
    _selftest()
