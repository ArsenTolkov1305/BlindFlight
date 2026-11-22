import math

import numpy as np
import pytest

from uahybrid.drift import DriftParams, r_err
from uahybrid.maps import Map, make_map
from uahybrid.planners.astar import astar
from uahybrid.sim import simulate_flight

ZERO = DriftParams(sigma_gps=0, sigma_beacon=0, drift_rate=0, walk_sigma=0)


def _tiny_map(blocked_cell=None):
    occ = np.zeros((20, 20), dtype=bool)
    if blocked_cell:
        occ[blocked_cell[1], blocked_cell[0]] = True
    gps = np.zeros((20, 20), dtype=bool)
    return Map(occ=occ, gps=gps, beacons=[], start=(0, 0), goal=(19, 0))


def test_no_noise_no_collision_on_free_path():
    m = _tiny_map()
    path = [(x, 5) for x in range(20)]
    res = simulate_flight(m, path, ZERO, np.random.default_rng(0))
    assert not res.collided and res.max_error == 0.0


def test_no_noise_path_through_obstacle_collides():
    m = _tiny_map(blocked_cell=(10, 5))
    path = [(x, 5) for x in range(20)]
    res = simulate_flight(m, path, ZERO, np.random.default_rng(0))
    assert res.collided and res.collision_step == 10


def test_same_rng_same_result():
    m = make_map(seed=3)
    path = astar(m.occ, m.start, m.goal)
    a = simulate_flight(m, path, DriftParams(), np.random.default_rng(7))
    b = simulate_flight(m, path, DriftParams(), np.random.default_rng(7))
    assert a.err == b.err and a.collided == b.collided


def test_r_err_grows_with_distance():
    p = DriftParams()
    vals = [r_err(s, p, p.sigma_gps) for s in (0, 10, 50, 100)]
    assert vals == sorted(vals) and vals[0] == pytest.approx(3 * p.sigma_gps)


def test_drift_is_larger_without_gps():
    m = _tiny_map()
    path = [(x, 5) for x in range(20)]
    res = simulate_flight(m, path, DriftParams(), np.random.default_rng(1), stop_on_collision=False)
    assert res.max_s == pytest.approx(19.0)


def test_bound_covers_error_in_almost_all_steps():
    m = make_map(seed=5, n_obstacles=25)
    path = astar(m.occ, m.start, m.goal)
    inside = total = 0
    for j in range(200):
        res = simulate_flight(m, path, DriftParams(), np.random.default_rng(j), stop_on_collision=False)
        inside += sum(e <= b for e, b in zip(res.err, res.bound))
        total += len(res.err)
    assert inside / total >= 0.97