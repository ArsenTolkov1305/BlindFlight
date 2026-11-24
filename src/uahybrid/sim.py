import math
from dataclasses import dataclass, field

import numpy as np

from uahybrid.drift import DriftParams, r_err


@dataclass
class FlightResult:
    collided: bool
    collision_step: int
    max_error: float
    max_s: float
    err: list = field(default_factory=list)
    bound: list = field(default_factory=list)
    collision_in_zone: bool = False


def simulate_flight(m, path, params: DriftParams, rng, stop_on_collision: bool = True,
                    k: float = 3.0, shield=None) -> FlightResult:
    occ, gps = m.occ, m.gps
    h, w = occ.shape
    n = len(path)

    walk = rng.normal(size=(n, 2))
    fix_gps = rng.normal(size=(n, 2)) * params.sigma_gps
    fix_beacon = rng.normal(size=(n, 2)) * params.sigma_beacon
    angle = rng.uniform(0, 2 * math.pi)
    bx, by = params.drift_rate * math.cos(angle), params.drift_rate * math.sin(angle)

    ex, ey = fix_gps[0]
    base, s = params.sigma_gps, 0.0
    res = FlightResult(False, -1, 0.0, 0.0)
    r2 = params.beacon_radius ** 2

    for i, (x, y) in enumerate(path):
        if i > 0:
            ds = math.hypot(x - path[i - 1][0], y - path[i - 1][1])
            sq = math.sqrt(ds)
            ex += bx * ds + params.walk_sigma * sq * walk[i, 0]
            ey += by * ds + params.walk_sigma * sq * walk[i, 1]
            s += ds
        tx, ty = x + ex, y + ey
        cx, cy = int(math.floor(tx + 0.5)), int(math.floor(ty + 0.5))

        if shield is not None and 0 <= cx < w and 0 <= cy < h and not occ[cy, cx]:
            px, py = shield.push(cx, cy)
            if px or py:
                ex, ey = ex + px, ey + py
                tx, ty = x + ex, y + ey
                cx, cy = int(math.floor(tx + 0.5)), int(math.floor(ty + 0.5))

        e = math.hypot(ex, ey)
        res.err.append(e)
        res.bound.append(r_err(s, params, base, k))
        res.max_error = max(res.max_error, e)
        res.max_s = max(res.max_s, s)

        if not (0 <= cx < w and 0 <= cy < h) or occ[cy, cx]:
            if res.collision_step < 0:
                res.collided, res.collision_step = True, i
                res.collision_in_zone = not gps[min(max(cy, 0), h - 1), min(max(cx, 0), w - 1)]
            if stop_on_collision:
                break
        cx, cy = min(max(cx, 0), w - 1), min(max(cy, 0), h - 1)

        if gps[cy, cx]:
            ex, ey = fix_gps[i]
            base, s = params.sigma_gps, 0.0
        elif any((tx - qx) ** 2 + (ty - qy) ** 2 <= r2 for qx, qy in m.beacons):
            ex, ey = fix_beacon[i]
            base, s = params.sigma_beacon, 0.0

    return res