import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from uahybrid.drift import DriftParams, r_err
from uahybrid.inflate import distance_map
from uahybrid.planners.astar import _NEIGHBORS, _octile


@dataclass(frozen=True)
class UAParams:
    r_max: float = 8.0
    lam: float = 0.5
    bucket: float = 4.0
    k: float = 3.0
    pass_radius: float = 1.5
    margin: float = 0.71


class _Ctx:

    def __init__(self, m, p: DriftParams, ua: UAParams, dist=None):
        self.p, self.ua, self.gps = p, ua, m.gps
        self.dist = distance_map(m.occ) if dist is None else dist
        if m.beacons:
            mask = np.zeros(m.occ.shape, dtype=bool)
            for x, y in m.beacons:
                mask[y, x] = True
            bdist = ndimage.distance_transform_edt(~mask)
        else:
            bdist = np.full(m.occ.shape, math.inf)
        self.pass_mask = bdist <= ua.pass_radius


def _transition(c: _Ctx, nx, ny, s, b, ds):
    base = c.p.sigma_beacon if b else c.p.sigma_gps
    s_pre = s + ds
    r_pre = r_err(s_pre, c.p, base, c.ua.k)
    if c.gps[ny, nx]:
        return r_pre, 0.0, 0
    if c.pass_mask[ny, nx] and r_pre + c.ua.pass_radius <= c.p.beacon_radius:
        return r_pre, 0.0, 1
    return r_pre, s_pre, b


def ua_astar(m, p: DriftParams = DriftParams(), ua: UAParams = UAParams(), dist=None):
    c = _Ctx(m, p, ua, dist)
    occ, dist = m.occ, c.dist
    height, width = occ.shape
    start, goal = tuple(m.start), tuple(m.goal)
    bucket = ua.bucket

    key0 = (start[0], start[1], 0, 0)
    g_best = {key0: 0.0}
    parent = {key0: None}
    closed = set()
    heap = [(_octile(start, goal), 0.0, start[0], start[1], 0.0, 0)]

    while heap:
        _, g, x, y, s, b = heapq.heappop(heap)
        key = (x, y, int(s / bucket), b)
        if key in closed or g > g_best[key] + 1e-9:
            continue
        if (x, y) == goal:
            path = []
            while key is not None:
                path.append((key[0], key[1]))
                key = parent[key]
            return path[::-1]
        closed.add(key)

        for dx, dy, ds in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height) or occ[ny, nx]:
                continue
            if dx and dy and (occ[y, nx] or occ[ny, x]):
                continue
            r_pre, s_new, b_new = _transition(c, nx, ny, s, b, ds)
            if r_pre > ua.r_max or dist[ny, nx] <= r_pre + ua.margin:
                continue
            nkey = (nx, ny, int(s_new / bucket), b_new)
            if nkey in closed:
                continue
            ng = g + ds * (1 + ua.lam * r_pre / ua.r_max)
            if ng < g_best.get(nkey, math.inf):
                g_best[nkey] = ng
                parent[nkey] = key
                heapq.heappush(heap, (ng + _octile((nx, ny), goal), ng, nx, ny, s_new, b_new))
    return None


def uncertainty_trace(m, path, p: DriftParams = DriftParams(), ua: UAParams = UAParams(), dist=None):
    c = _Ctx(m, p, ua, dist)
    s, b, out = 0.0, 0, [0.0]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        r_pre, s, b = _transition(c, x1, y1, s, b, math.hypot(x1 - x0, y1 - y0))
        out.append(r_pre)
    return out