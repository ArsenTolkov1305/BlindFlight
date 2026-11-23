import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
from tqdm import tqdm

from uahybrid.drift import DriftParams
from uahybrid.inflate import distance_map, inflate
from uahybrid.maps import make_map
from uahybrid.metrics import path_length
from uahybrid.planners.astar import astar
from uahybrid.planners.ua_astar import UAParams
from uahybrid.planners.ua_astar_rs import HAVE_RUST, ua_astar_fast
from uahybrid.sim import simulate_flight

DRIFT_RATES = [0.02, 0.04, 0.08]
STATIC_R = {"A* r=0": 0, "static r=4": 4, "static r=6": 6, "static r=8": 8}
N_FLIGHTS = 20
UA = UAParams()


def run_seed(seed):
    m = make_map(size=200, n_obstacles=25, seed=seed)
    dist = distance_map(m.occ)
    static_paths = {name: astar(inflate(m.occ, r, dist), m.start, m.goal) for name, r in STATIC_R.items()}
    touches = any(not m.gps[y, x] for x, y in static_paths["A* r=0"])
    rows = []
    for dr in DRIFT_RATES:
        p = replace(DriftParams(), drift_rate=dr)
        paths = dict(static_paths)
        paths["UA-A*"] = ua_astar_fast(m, p, UA, dist)
        for name, path in paths.items():
            row = {"seed": seed, "drift_rate": dr, "method": name,
                   "touches_zone": touches, "planned": path is not None}
            if path is not None:
                crashes = sum(
                    simulate_flight(m, path, p, np.random.default_rng(seed * 1000 + j)).collided
                    for j in range(N_FLIGHTS))
                row.update(length=path_length(path), crash_rate=crashes / N_FLIGHTS)
            rows.append(row)
    return rows


def summarize(df):
    d = df[df.touches_zone].copy()
    d["mission_ok"] = np.where(d.planned, 1 - d.crash_rate.fillna(1), 0.0)
    base_len = d[d.method == "A* r=0"].set_index(["seed", "drift_rate"]).length
    d["length_ratio"] = d.length.values / base_len.reindex(list(zip(d.seed, d.drift_rate))).values
    order = list(STATIC_R) + ["UA-A*"]
    print(f"Карт, де A* заходить у зону без GPS: {d.seed.nunique()} з {df.seed.nunique()}; "
          f"польотів на план: {N_FLIGHTS}")
    for dr in DRIFT_RATES:
        x = d[d.drift_rate == dr]
        common = x.groupby("seed").planned.all()
        x = x.assign(in_common=x.seed.map(common))
        t = x.groupby("method").agg(planned=("planned", "mean"), crash=("crash_rate", "mean"),
                                    mission_ok=("mission_ok", "mean")).round(2)
        t["length_ratio"] = x[x.in_common].groupby("method").length_ratio.mean().round(2)
        print(f"\nдрейф {dr:.2f} (довжина - лише на {int(common.sum())} картах, де спланували всі)")
        print(t.reindex(order))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=100, help="скільки карт (seed 0..N-1)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="кількість процесів")
    ap.add_argument("--out", default="results/compare.csv")
    ap.add_argument("--first-seed", type=int, default=0)
    args = ap.parse_args()

    seeds = range(args.first_seed, args.first_seed + args.maps)
    print("UA-A* backend:", "Rust" if HAVE_RUST else "Python")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(tqdm(pool.map(run_seed, seeds), total=len(seeds)))
    else:
        results = [run_seed(s) for s in tqdm(seeds)]

    df = pd.DataFrame([row for rows in results for row in rows])
    df.to_csv(args.out, index=False)
    summarize(df)


if __name__ == "__main__":
    main()