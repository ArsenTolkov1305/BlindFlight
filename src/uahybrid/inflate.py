import math

import numpy as np
from scipy import ndimage


def distance_map(occ: np.ndarray) -> np.ndarray:
    if not occ.any():
        return np.full(occ.shape, math.inf)
    return ndimage.distance_transform_edt(~occ)


def inflate(occ: np.ndarray, r: float, dist: np.ndarray = None) -> np.ndarray:
    if dist is None:
        dist = distance_map(occ)
    return dist <= r