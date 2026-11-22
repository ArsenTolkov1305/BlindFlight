import math

import numpy as np
from scipy import ndimage


def path_length(path) -> float:
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])))


def min_clearance(path, occ: np.ndarray) -> float:
    if not occ.any():
        return math.inf
    dist = ndimage.distance_transform_edt(~occ)
    return float(min(dist[y, x] for x, y in path))