import numpy as np
import pandas as pd
from tqdm import tqdm

from uahybrid.drift import DriftParams
from uahybrid.inflate import distance_map, inflate
from uahybrid.maps import make_map
from uahybrid.metrics import path_length
from uahybrid.planners.astar import astar
from uahybrid.sim import simulate_flight

SEEDS = range(100)
RADII = [0, 2, 4, 6, 8, 10]
N_FLIGHTS = 20
PARAMS = DriftParams()

rows = []
for seed in tqdm(SEEDS):
    m = make_map(size=200, n_obstacles=25, seed=seed)
    dist = distance_map(m.occ)
    for r in RADII:
        path = astar(inflate(m.occ, r, dist), m.start, m.goal)
        if path is None:
            rows.append({"seed": seed, "r": r, "planned": False})
            continue
        crashes = sum(
            simulate_flight(m, path, PARAMS, np.random.default_rng(seed * 1000 + j)).collided
            for j in range(N_FLIGHTS)
        )
        rows.append({
            "seed": seed, "r": r, "planned": True,
            "length": path_length(path),
            "denied_share": float(np.mean([not m.gps[y, x] for x, y in path])),
            "crash_rate": crashes / N_FLIGHTS,
        })

df = pd.DataFrame(rows)
df.to_csv("results/drift_baseline.csv", index=False)

base = df[df.r == 0].set_index("seed")
relevant = base.index[base.denied_share > 0]
d = df[df.seed.isin(relevant)].copy()
d["length_ratio"] = d.length / d.seed.map(base.length)
d["mission_ok"] = np.where(d.planned, 1 - d.crash_rate, 0.0)

print(f"Карт, де A* заходить у зону без GPS: {len(relevant)} з {len(SEEDS)}")
print(d.groupby("r").agg(
    planned=("planned", "mean"),
    crash_rate=("crash_rate", "mean"),
    mission_ok=("mission_ok", "mean"),
    length_ratio=("length_ratio", "mean"),
).round(2))