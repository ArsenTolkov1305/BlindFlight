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
from uahybrid.shield import Shield, ShieldParams
from uahybrid.sim import simulate_flight

N_FLIGHTS = 20
PUSHES = [0.0, 0.1, 0.3, 1.0]
METHODS = ["A* r=0", "static r=6", "UA-A*"]
PARAMS = DriftParams()
UA = UAParams()


def run_seed(seed, drift_rate):
    p = replace(PARAMS, drift_rate=drift_rate)
    m = make_map(size=200, n_obstacles=25, seed=seed)
    dist = distance_map(m.occ)
    paths = {
        "A* r=0": astar(m.occ, m.start, m.goal),
        "static r=6": astar(inflate(m.occ, 6, dist), m.start, m.goal),
        "UA-A*": ua_astar_fast(m, p, UA, dist),
    }
    touches = any(not m.gps[y, x] for x, y in paths["A* r=0"])
    shields = {push: Shield(m.occ, ShieldParams(max_push=push), dist) for push in PUSHES if push > 0}
    rows = []
    for name, path in paths.items():
        for push in PUSHES:
            row = {"seed": seed, "method": name, "max_push": push,
                   "touches_zone": touches, "planned": path is not None}
            if path is not None:
                flights = [simulate_flight(m, path, p, np.random.default_rng(seed * 1000 + j),
                                           shield=shields.get(push)) for j in range(N_FLIGHTS)]
                row.update(length=path_length(path),
                           crash_rate=sum(f.collided for f in flights) / N_FLIGHTS,
                           zone_crash_rate=sum(f.collision_in_zone for f in flights) / N_FLIGHTS)
            rows.append(row)
    return rows


def _job(args):
    return run_seed(*args)


def summarize(df, drift_rate):
    d = df[df.touches_zone].copy()
    d["mission_ok"] = np.where(d.planned, 1 - d.crash_rate.fillna(1), 0.0)
    print(f"Дрейф {drift_rate}; карт, де A* заходить у зону без GPS: {d.seed.nunique()} з {df.seed.nunique()}; "
          f"польотів на план: {N_FLIGHTS}")
    for col, title in [("crash_rate", "Частка аварій (серед спланованих), усього"),
                       ("zone_crash_rate", "з них аварії САМЕ в зоні без GPS (там накопичується дрейф)"),
                       ("mission_ok", "Доліт (спланував і долетів)")]:
        t = d.pivot_table(index="method", columns="max_push", values=col, aggfunc="mean").round(2)
        t.columns = ["без щита" if c == 0 else f"щит {c}" for c in t.columns]
        print(f"\n{title}")
        print(t.reindex(METHODS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=PARAMS.drift_rate)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/shield.csv")
    args = ap.parse_args()

    jobs = [(s, args.drift) for s in range(args.first_seed, args.first_seed + args.maps)]
    print("UA-A* backend:", "Rust" if HAVE_RUST else "Python")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(tqdm(pool.map(_job, jobs), total=len(jobs)))
    else:
        results = [_job(j) for j in tqdm(jobs)]

    df = pd.DataFrame([row for rows in results for row in rows])
    df.to_csv(args.out, index=False)
    summarize(df, args.drift)


if __name__ == "__main__":
    main()