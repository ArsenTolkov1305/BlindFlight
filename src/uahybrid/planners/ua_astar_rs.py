import numpy as np

from uahybrid.drift import DriftParams
from uahybrid.planners.ua_astar import UAParams, _Ctx, ua_astar

try:
    import ua_core
    HAVE_RUST = hasattr(ua_core, "ua_astar")
except ImportError:
    ua_core = None
    HAVE_RUST = False


def ua_astar_rs(m, p: DriftParams = DriftParams(), ua: UAParams = UAParams(), dist=None):
    if not HAVE_RUST:
        raise RuntimeError("ua_core не зібраний: cd rust/ua_core && maturin develop --release")
    c = _Ctx(m, p, ua, dist)
    height, width = m.occ.shape
    params = [p.sigma_gps, p.sigma_beacon, p.drift_rate, p.walk_sigma, p.beacon_radius,
              ua.r_max, ua.lam, ua.bucket, ua.bucket_r, ua.k, ua.pass_radius, ua.margin]
    return ua_core.ua_astar(
        width, height,
        np.ascontiguousarray(m.occ, dtype=np.uint8).tobytes(),
        np.ascontiguousarray(m.gps, dtype=np.uint8).tobytes(),
        np.ascontiguousarray(c.pass_mask, dtype=np.uint8).tobytes(),
        np.ascontiguousarray(c.dist, dtype="<f8").tobytes(),
        (int(m.start[0]), int(m.start[1])), (int(m.goal[0]), int(m.goal[1])),
        params,
    )


def ua_astar_fast(m, p: DriftParams = DriftParams(), ua: UAParams = UAParams(), dist=None):
    if HAVE_RUST and ua.dynamic:
        return ua_astar_rs(m, p, ua, dist)
    return ua_astar(m, p, ua, dist)