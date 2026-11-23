import time

import numpy as np

from uahybrid.inflate import distance_map
from uahybrid.maps import make_map
from uahybrid.metrics import path_length
from uahybrid.planners.ua_astar import ua_astar
from uahybrid.planners.ua_astar_rs import HAVE_RUST, ua_astar_rs

if not HAVE_RUST:
    raise SystemExit("ua_core не зібраний: cd rust/ua_core && maturin develop --release")

maps = [make_map(size=200, n_obstacles=25, seed=s) for s in range(20)]
dists = [distance_map(m.occ) for m in maps]

t = time.perf_counter()
py = [ua_astar(m, dist=d) for m, d in zip(maps, dists)]
t_py = time.perf_counter() - t

t = time.perf_counter()
rs = [ua_astar_rs(m, dist=d) for m, d in zip(maps, dists)]
t_rs = time.perf_counter() - t

same = sum(a == b for a, b in zip(py, rs))
same_len = sum(abs(path_length(a) - path_length(b)) < 1e-9 for a, b in zip(py, rs))
print(f"Python: {1000 * t_py / len(maps):.1f} мс на план")
print(f"Rust:   {1000 * t_rs / len(maps):.2f} мс на план  (у {t_py / t_rs:.0f} разів швидше)")
print(f"Шляхи повністю збігаються: {same}/{len(maps)}; однакова довжина: {same_len}/{len(maps)}")