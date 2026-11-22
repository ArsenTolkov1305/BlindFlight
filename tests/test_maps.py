import numpy as np
import pytest
from scipy import ndimage

from uahybrid.maps import make_map


def test_same_seed_same_map():
    m1 = make_map(seed=1)
    m2 = make_map(seed=1)

    assert np.array_equal(m1.occ, m2.occ), "Obstacle grids (occ) do not match for the same seed"
    assert np.array_equal(m1.gps, m2.gps), "GPS grids do not match for the same seed"
    assert m1.start == m2.start, "Start points do not match for the same seed"
    assert m1.goal == m2.goal, "Goal points do not match for the same seed"
    assert m1.beacons == m2.beacons, "Beacons do not match for the same seed"


@pytest.mark.parametrize("seed", range(20))
def test_points_are_free(seed):
    m = make_map(seed=seed)
    all_points = [m.start, m.goal] + m.beacons

    for x, y in all_points:
        assert not m.occ[y, x], f"Point ({x}, {y}) is placed inside an obstacle"


def test_start_goal_have_gps():
    m = make_map(seed=42)

    for x, y in [m.start, m.goal]:
        assert m.gps[y, x], f"GPS signal is missing at ({x}, {y})"


def test_shapes_and_beacons():
    size = 150
    m = make_map(size=size, seed=42)

    assert m.occ.shape == (size, size), f"Expected occ shape {(size, size)}, got {m.occ.shape}"
    assert m.gps.shape == (size, size), f"Expected gps shape {(size, size)}, got {m.gps.shape}"
    assert len(m.beacons) == 5, f"Expected 5 beacons, got {len(m.beacons)}"


def test_some_beacons_in_gps_denied_zone():
    m = make_map(seed=42)
    in_zone = [b for b in m.beacons if not m.gps[b[1], b[0]]]
    assert len(in_zone) >= 3, f"Expected at least 3 beacons in GPS-denied zone, got {len(in_zone)}"


@pytest.mark.parametrize("seed", range(20))
def test_start_goal_distance(seed):
    size = 200
    m = make_map(size=size, seed=seed)
    dist = np.hypot(m.goal[0] - m.start[0], m.goal[1] - m.start[1])

    assert dist >= 0.7 * size, f"Distance {dist:.2f} is less than required {0.7 * size}"


@pytest.mark.parametrize("seed", range(20))
def test_start_left_goal_right(seed):
    size = 200
    m = make_map(size=size, seed=seed)
    assert m.start[0] < size * 0.15
    assert m.goal[0] >= size * 0.85


@pytest.mark.parametrize("seed", range(20))
def test_start_and_goal_connected(seed):
    m = make_map(seed=seed)
    labels, _ = ndimage.label(~m.occ)
    assert labels[m.start[1], m.start[0]] == labels[m.goal[1], m.goal[0]]


@pytest.mark.parametrize("seed", range(20))
def test_beacons_are_distinct(seed):
    m = make_map(seed=seed)
    assert len(set(m.beacons)) == len(m.beacons)


def test_small_map_works():
    m = make_map(size=80, seed=3)
    assert m.occ.shape == (80, 80)


def test_too_small_map_raises():
    with pytest.raises(ValueError):
        make_map(size=30)


@pytest.mark.parametrize("seed", range(20))
def test_start_goal_have_free_space_around(seed):
    from uahybrid.inflate import distance_map
    size = 200
    m = make_map(size=size, seed=seed)
    dist = distance_map(m.occ)
    assert dist[m.start[1], m.start[0]] >= size * 0.06
    assert dist[m.goal[1], m.goal[0]] >= size * 0.06