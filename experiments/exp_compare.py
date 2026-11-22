from dataclasses import replace

import numpy as np
import pandas as pd
from tqdm import tqdm

from uahybrid.drift import DriftParams
from uahybrid.inflate import distance_map, inflate
from uahybrid.maps import make_map
from uahybrid.metrics import path_length
from uahybrid.planners.astar import astar
from uahybrid.planners.ua_astar import UAParams, ua_astar
from uahybrid.sim import simulate_flight

SEEDS = range(100)
DRIFT_RATES = [0.02, 0.04, 0.08]
STATIC_R = {"A* r=0": 0, "static r=4": 4, "static r=6": 6, "static r=8": 8}
N_FLIGHTS = 20
UA = UAParams()

rows = []
for seed in tqdm(SEEDS):
    m = make_map(size=200, n_obstacles=25, seed=seed)
    dist = distance_map(m.occ)
    static_paths = {name: astar(inflate(m.occ, r, dist), m.start, m.goal) for name, r in STATIC_R.items()}
    base_path = static_paths["A* r=0"]
    touches = any(not m.gps[y, x] for x, y in base_path)
    for dr in DRIFT_RATES:
        p = replace(DriftParams(), drift_rate=dr)
        paths = dict(static_paths)
        paths["UA-A*"] = ua_astar(m, p, UA, dist)
        for name, path in paths.items():
            row = {"seed": seed, "drift_rate": dr, "method": name,
                   "touches_zone": touches, "planned": path is not None}
            if path is not None:
                crashes = sum(
                    simulate_flight(m, path, p, np.random.default_rng(seed * 1000 + j)).collided
                    for j in range(N_FLIGHTS))
                row.update(length=path_length(path), crash_rate=crashes / N_FLIGHTS)
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("results/compare.csv", index=False)

d = df[df.touches_zone].copy()
n_maps = d.seed.nunique()
d["mission_ok"] = np.where(d.planned, 1 - d.crash_rate.fillna(1), 0.0)
base_len = d[d.method == "A* r=0"].set_index(["seed", "drift_rate"]).length
d["length_ratio"] = d.length.values / base_len.reindex(list(zip(d.seed, d.drift_rate))).values

order = list(STATIC_R) + ["UA-A*"]
print(f"Карт, де A* заходить у зону без GPS: {n_maps} з {len(SEEDS)}; польотів на план: {N_FLIGHTS}")
for dr in DRIFT_RATES:
    x = d[d.drift_rate == dr]
    common = x.groupby("seed").planned.all()
    x = x.assign(in_common=x.seed.map(common))
    t = x.groupby("method").agg(planned=("planned", "mean"), crash=("crash_rate", "mean"),
                                mission_ok=("mission_ok", "mean")).round(2)
    t["length_ratio"] = x[x.in_common].groupby("method").length_ratio.mean().round(2)
    print(f"\nдрейф {dr:.2f} (довжина - лише на {int(common.sum())} картах, де спланували всі)")
    print(t.reindex(order))