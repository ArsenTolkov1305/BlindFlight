import numpy as np
import pytest

from uahybrid.inflate import distance_map
from uahybrid.maps import Map, make_map
from uahybrid.planners.ua_astar import UAParams, ua_astar, uncertainty_trace


def _open_map(width=100, beacons=()):
    occ = np.zeros((40, width), dtype=bool)
    gps = np.zeros((40, width), dtype=bool)
    gps[:, :10] = True
    return Map(occ=occ, gps=gps, beacons=list(beacons), start=(2, 20), goal=(width - 12, 20))


def test_too_far_without_gps_or_beacons_returns_none():
    assert ua_astar(_open_map()) is None


def test_beacon_makes_long_flight_possible():
    m = _open_map(beacons=[(45, 20)])
    path = ua_astar(m)
    assert path is not None
    assert path[0] == m.start and path[-1] == m.goal
    assert min(np.hypot(x - 45, y - 20) for x, y in path) <= UAParams().pass_radius


def test_gps_everywhere_keeps_margin():
    m = make_map(seed=4, n_obstacles=25)
    m.gps[:] = True
    path = ua_astar(m)
    dist = distance_map(m.occ)
    assert path is not None
    assert all(dist[y, x] > 1.5 for x, y in path[1:])


@pytest.mark.parametrize("seed", range(10))
def test_path_respects_own_uncertainty_model(seed):
    m = make_map(seed=seed, n_obstacles=25)
    path = ua_astar(m)
    if path is None:
        pytest.skip("no safe path on this map")
    ua = UAParams()
    dist = distance_map(m.occ)
    trace = uncertainty_trace(m, path)
    assert path[0] == m.start and path[-1] == m.goal
    for (x, y), r in list(zip(path, trace))[1:]:
        assert r <= ua.r_max + 1e-9
        assert dist[y, x] > r + ua.margin