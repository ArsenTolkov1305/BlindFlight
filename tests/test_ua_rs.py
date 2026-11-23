from dataclasses import replace

import numpy as np
import pytest

from uahybrid.drift import DriftParams
from uahybrid.maps import Map, make_map
from uahybrid.metrics import path_length
from uahybrid.planners.ua_astar import ua_astar
from uahybrid.planners.ua_astar_rs import HAVE_RUST, ua_astar_rs

if not HAVE_RUST:
    pytest.skip("ua_core не зібраний", allow_module_level=True)


@pytest.mark.parametrize("seed", range(30))
def test_rust_matches_python(seed):
    m = make_map(seed=seed, n_obstacles=25)
    py, rs = ua_astar(m), ua_astar_rs(m)
    assert (py is None) == (rs is None)
    if py is not None:
        assert rs[0] == m.start and rs[-1] == m.goal
        assert path_length(rs) == pytest.approx(path_length(py), rel=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_rust_matches_python_strong_drift(seed):
    p = replace(DriftParams(), drift_rate=0.08)
    m = make_map(seed=seed, n_obstacles=25)
    py, rs = ua_astar(m, p), ua_astar_rs(m, p)
    assert (py is None) == (rs is None)
    if py is not None:
        assert path_length(rs) == pytest.approx(path_length(py), rel=1e-9)


def test_rust_returns_none_when_too_far():
    occ = np.zeros((40, 100), dtype=bool)
    gps = np.zeros((40, 100), dtype=bool)
    gps[:, :10] = True
    m = Map(occ=occ, gps=gps, beacons=[], start=(2, 20), goal=(88, 20))
    assert ua_astar_rs(m) is None


def test_rust_uses_beacon():
    occ = np.zeros((40, 100), dtype=bool)
    gps = np.zeros((40, 100), dtype=bool)
    gps[:, :10] = True
    m = Map(occ=occ, gps=gps, beacons=[(45, 20)], start=(2, 20), goal=(88, 20))
    path = ua_astar_rs(m)
    assert path is not None
    assert min(np.hypot(x - 45, y - 20) for x, y in path) <= 1.5