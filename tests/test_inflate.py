import numpy as np
import pytest

from uahybrid.inflate import distance_map, inflate


def _single_obstacle():
    occ = np.zeros((11, 11), dtype=bool)
    occ[5, 5] = True
    return occ


def test_zero_radius_equals_occ():
    occ = _single_obstacle()
    assert np.array_equal(inflate(occ, 0), occ)


def test_radius_two_blocks_disk():
    occ = _single_obstacle()
    blocked = inflate(occ, 2)
    assert blocked[5, 7] and blocked[7, 5]
    assert blocked[6, 6]
    assert not blocked[5, 8]
    assert not blocked[7, 7]


@pytest.mark.parametrize("r1,r2", [(1, 2), (2, 5), (0, 3)])
def test_inflation_is_monotonic(r1, r2):
    occ = _single_obstacle()
    small, big = inflate(occ, r1), inflate(occ, r2)
    assert not np.any(small & ~big)


def test_no_obstacles_nothing_blocked():
    occ = np.zeros((6, 6), dtype=bool)
    assert not inflate(occ, 3).any()


def test_precomputed_distance_gives_same_result():
    occ = _single_obstacle()
    d = distance_map(occ)
    assert np.array_equal(inflate(occ, 2, d), inflate(occ, 2))