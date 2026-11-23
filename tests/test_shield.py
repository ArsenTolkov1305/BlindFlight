import numpy as np
import pytest

from uahybrid.drift import DriftParams
from uahybrid.maps import make_map
from uahybrid.planners.astar import astar
from uahybrid.shield import Shield, ShieldParams
from uahybrid.sim import simulate_flight


def _wall_occ():
    occ = np.zeros((20, 20), dtype=bool)
    occ[:, 10:] = True
    return occ


def test_push_is_away_from_wall():
    sh = Shield(_wall_occ(), ShieldParams(d_safe=2.0, max_push=0.3))
    px, py = sh.push(9, 5)
    assert px < 0 and abs(py) < 1e-6
    assert np.hypot(px, py) == pytest.approx(0.3)


def test_no_push_when_far_or_disabled():
    occ = _wall_occ()
    assert Shield(occ).push(3, 5) == (0.0, 0.0)
    assert Shield(occ, ShieldParams(max_push=0.0)).push(9, 5) == (0.0, 0.0)


def test_push_never_exceeds_distance_to_safe_line():
    sh = Shield(_wall_occ(), ShieldParams(d_safe=2.0, max_push=5.0))
    px, py = sh.push(9, 5)
    assert np.hypot(px, py) == pytest.approx(1.0)


def test_zero_max_push_equals_no_shield():
    m = make_map(seed=3, n_obstacles=25)
    path = astar(m.occ, m.start, m.goal)
    a = simulate_flight(m, path, DriftParams(), np.random.default_rng(5))
    b = simulate_flight(m, path, DriftParams(), np.random.default_rng(5),
                        shield=Shield(m.occ, ShieldParams(max_push=0.0)))
    assert a.err == b.err and a.collided == b.collided


def test_shield_reduces_crashes_on_hugging_path():
    crashes_without = crashes_with = 0
    for seed in range(6):
        m = make_map(seed=seed, n_obstacles=25)
        path = astar(m.occ, m.start, m.goal)
        sh = Shield(m.occ, ShieldParams(d_safe=2.0, max_push=1.0))
        for j in range(20):
            p = DriftParams(drift_rate=0.08)
            crashes_without += simulate_flight(m, path, p, np.random.default_rng(j)).collided
            crashes_with += simulate_flight(m, path, p, np.random.default_rng(j), shield=sh).collided
    assert crashes_with < crashes_without