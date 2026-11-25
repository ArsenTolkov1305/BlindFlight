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

SCENARIOS = {"random": None, "band80": 80, "band110": 110}
DRIFTS = [0.04, 0.08]
N_FLIGHTS = 20
N_BEACONS_BAND = 6
METHODS = ["A* r=0", "static r=6", "UA-A* повний", "без правила 1", "без правила 2", "без правила 3"]
FULL = UAParams(r_max=15.0)


def make_scenario(name, seed):
    m = make_map(size=200, n_obstacles=25, seed=seed)
    width = SCENARIOS[name]
    if width is None:
        return m
    size = m.occ.shape[0]
    gps = np.ones_like(m.gps)
    x0 = (size - width) // 2
    gps[:, x0:x0 + width] = False
    ys, xs = np.where(~m.occ & ~gps)
    rng = np.random.default_rng(10_000 + seed)
    idx = rng.choice(len(xs), size=N_BEACONS_BAND, replace=False)
    beacons = [(int(xs[i]), int(ys[i])) for i in idx]
    return replace(m, gps=gps, beacons=beacons)


def run_job(args):
    scenario, seed = args
    m = make_scenario(scenario, seed)
    dist = distance_map(m.occ)
    m_blind = replace(m, beacons=[])                      # планувальник без маяків
    base = {"A* r=0": astar(m.occ, m.start, m.goal),
            "static r=6": astar(inflate(m.occ, 6, dist), m.start, m.goal)}
    touches = any(not m.gps[y, x] for x, y in base["A* r=0"])
    rows = []
    for dr in DRIFTS:
        p = replace(DriftParams(), drift_rate=dr)
        paths = dict(base)
        paths["UA-A* повний"] = ua_astar_fast(m, p, FULL, dist)
        paths["без правила 1"] = ua_astar_fast(m, p, replace(FULL, dynamic=False), dist)
        paths["без правила 2"] = ua_astar_fast(m, p, replace(FULL, lam=0.0), dist)
        paths["без правила 3"] = ua_astar_fast(m_blind, p, FULL, dist)
        for name, path in paths.items():
            row = {"scenario": scenario, "seed": seed, "drift": dr, "method": name,
                   "touches_zone": touches, "planned": path is not None}
            if path is not None:
                flights = [simulate_flight(m, path, p, np.random.default_rng(seed * 1000 + j))
                           for j in range(N_FLIGHTS)]
                row.update(length=path_length(path),
                           crash_rate=sum(f.collided for f in flights) / N_FLIGHTS,
                           zone_crash_rate=sum(f.collision_in_zone for f in flights) / N_FLIGHTS)
            rows.append(row)
    return rows


def summarize(df):
    for scenario in SCENARIOS:
        d = df[df.scenario == scenario]
        if scenario == "random":
            d = d[d.touches_zone]
        n = d.seed.nunique()
        for dr in DRIFTS:
            x = d[d.drift == dr].copy()
            x["mission_ok"] = np.where(x.planned, 1 - x.crash_rate.fillna(1), 0.0)
            ref = x[x.method == "A* r=0"].set_index("seed").length
            x["length_ratio"] = x.length.values / ref.reindex(x.seed).values
            t = x.groupby("method").agg(planned=("planned", "mean"), crash=("crash_rate", "mean"),
                                        mission_ok=("mission_ok", "mean"),
                                        length_ratio=("length_ratio", "mean")).round(2)
            print(f"\n=== {scenario}, дрейф {dr}, карт: {n}, польотів на план: {N_FLIGHTS} ===")
            print(t.reindex(METHODS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/ablation.csv")
    args = ap.parse_args()

    jobs = [(sc, s) for sc in SCENARIOS for s in range(args.first_seed, args.first_seed + args.maps)]
    print("UA-A* backend:", "Rust" if HAVE_RUST else "Python", "(правило 1 вимкнене рахується на Python)")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(tqdm(pool.map(run_job, jobs), total=len(jobs)))
    else:
        results = [run_job(j) for j in tqdm(jobs)]
    df = pd.DataFrame([row for rows in results for row in rows])
    df.to_csv(args.out, index=False)
    summarize(df)


if __name__ == "__main__":
    main()