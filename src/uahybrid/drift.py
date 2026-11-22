import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftParams:
    sigma_gps: float = 0.5
    sigma_beacon: float = 1.0
    drift_rate: float = 0.04
    walk_sigma: float = 0.15
    beacon_radius: float = 4.0


def r_err(s: float, p: DriftParams, base_sigma: float, k: float = 3.0) -> float:
    return k * base_sigma + p.drift_rate * s + k * p.walk_sigma * math.sqrt(s)