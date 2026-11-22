import math

import numpy as np
import pytest

from uahybrid.maps import make_map
from uahybrid.metrics import min_clearance, path_length
from uahybrid.planners.astar import astar


def test_empty_map_diagonal():
    blocked = np.zeros((10, 10), dtype=bool)
    path = astar(blocked, (0, 0), (9, 9))
    assert path[0] == (0, 0) and path[-1] == (9, 9)
    assert path_length(path) == pytest.approx(9 * math.sqrt(2))


def test_wall_with_single_gap():
    blocked = np.zeros((20, 20), dtype=bool)
    blocked[:, 10] = True          # стіна на x = 10
    blocked[15, 10] = False        # прохід на y = 15
    path = astar(blocked, (0, 0), (19, 0))
    assert path is not None
    assert (10, 15) in path
    assert all(not blocked[y, x] for x, y in path)


def test_unreachable_goal_returns_none():
    blocked = np.zeros((10, 10), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if (dx, dy) != (0, 0):
                blocked[5 + dy, 5 + dx] = True
    assert astar(blocked, (0, 0), (5, 5)) is None


def test_start_equals_goal():
    blocked = np.zeros((5, 5), dtype=bool)
    assert astar(blocked, (2, 2), (2, 2)) == [(2, 2)]


def test_no_corner_cutting():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[0, 1] = True           # клітинка (x=1, y=0)
    blocked[1, 0] = True           # клітинка (x=0, y=1)
    # діагональ (0,0) -> (1,1) заборонена, а інших виходів зі старту немає
    assert astar(blocked, (0, 0), (1, 1)) is None


def test_blocked_start_or_goal_returns_none():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[0, 0] = True
    assert astar(blocked, (0, 0), (4, 4)) is None
    assert astar(blocked, (4, 4), (0, 0)) is None


@pytest.mark.parametrize("seed", range(20))
def test_path_on_generated_maps(seed):
    m = make_map(seed=seed)
    path = astar(m.occ, m.start, m.goal)
    assert path is not None, f"No path on seed {seed}"
    assert path[0] == m.start and path[-1] == m.goal
    assert all(not m.occ[y, x] for x, y in path)
    # сусідні точки шляху - сусіди по сітці (крок не більше 1 по кожній осі)
    assert all(max(abs(b[0] - a[0]), abs(b[1] - a[1])) == 1 for a, b in zip(path, path[1:]))
    straight = math.hypot(m.goal[0] - m.start[0], m.goal[1] - m.start[1])
    assert path_length(path) >= straight - 1e-9


def test_min_clearance_simple():
    occ = np.zeros((10, 10), dtype=bool)
    occ[5, 5] = True
    assert min_clearance([(5, 2)], occ) == pytest.approx(3.0)
    assert min_clearance([(0, 0)], np.zeros((4, 4), dtype=bool)) == math.inf