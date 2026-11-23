import math
from dataclasses import dataclass

import numpy as np

from uahybrid.inflate import distance_map


@dataclass(frozen=True)
class ShieldParams:
    d_safe: float = 2.0
    max_push: float = 0.3


class Shield:
    def __init__(self, occ, params: ShieldParams = ShieldParams(), dist=None):
        self.p = params
        self.dist = distance_map(occ) if dist is None else dist
        finite = np.where(np.isfinite(self.dist), self.dist, 0.0)
        self.gy, self.gx = np.gradient(finite)

    def push(self, cx: int, cy: int):
        d = self.dist[cy, cx]
        if self.p.max_push <= 0 or d >= self.p.d_safe or d == 0:
            return 0.0, 0.0
        gx, gy = self.gx[cy, cx], self.gy[cy, cx]
        norm = math.hypot(gx, gy)
        if norm < 1e-9:
            return 0.0, 0.0
        amount = min(self.p.max_push, self.p.d_safe - d)
        return amount * gx / norm, amount * gy / norm