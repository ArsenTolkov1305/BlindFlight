import numpy as np
import pandas as pd
from tqdm import tqdm

from uahybrid.inflate import distance_map, inflate
from uahybrid.maps import make_map
from uahybrid.metrics import min_clearance, path_length
from uahybrid.planners.astar import astar

SEEDS = range(100)
RADII = [0, 2, 4, 6, 8, 10]

rows = []
for seed in tqdm(SEEDS):
    m = make_map(size=200, n_obstacles=25, seed=seed)
    dist = distance_map(m.occ)
    for r in RADII:
        path = astar(inflate(m.occ, r, dist), m.start, m.goal)
        ok = path is not None
        rows.append({
            "seed": seed,
            "r": r,
            "success": ok,
            "length": path_length(path) if ok else np.nan,
            "clearance": min_clearance(path, m.occ) if ok else np.nan,
        })

df = pd.DataFrame(rows)
df.to_csv("results/static_inflation.csv", index=False)

base = df[df.r == 0].set_index("seed")["length"]
df["length_ratio"] = df.apply(lambda row: row.length / base[row.seed], axis=1)

summary = df.groupby("r").agg(
    success_rate=("success", "mean"),
    length_ratio=("length_ratio", "mean"),
    clearance=("clearance", "mean"),
).round(2)
print(summary)