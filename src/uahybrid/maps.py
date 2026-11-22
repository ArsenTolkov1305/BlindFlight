from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


@dataclass
class Map:
    occ: np.ndarray
    gps: np.ndarray
    beacons: list
    start: tuple
    goal: tuple


def _pick(rng, mask, n):
    ys, xs = np.where(mask)
    if len(xs) < n:
        raise RuntimeError(f"Not enough free cells: need {n}, have {len(xs)}")
    idx = rng.choice(len(xs), size=n, replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def _try_make(rng, size, n_obstacles, n_zones):
    occ = np.zeros((size, size), dtype=bool)
    gps = np.ones((size, size), dtype=bool)

    for _ in range(n_obstacles):
        w, h = rng.integers(5, 26, size=2)
        x = rng.integers(0, size - w)
        y = rng.integers(0, size - h)
        occ[y:y + h, x:x + w] = True

    lo, hi = size // 4, int(size * 0.4)
    for _ in range(n_zones):
        w, h = rng.integers(lo, hi + 1, size=2)
        x = rng.integers(size // 4, min(size // 2, size - w) + 1)
        y = rng.integers(size // 4, min(size // 2, size - h) + 1)
        gps[y:y + h, x:x + w] = False

    strip = max(1, int(size * 0.15))
    free = ~occ & gps
    left = free.copy()
    left[:, strip:] = False
    right = free.copy()
    right[:, :size - strip] = False
    if not left.any() or not right.any():
        return None
    start = _pick(rng, left, 1)[0]
    goal = _pick(rng, right, 1)[0]

    labels, _ = ndimage.label(~occ)
    if labels[start[1], start[0]] != labels[goal[1], goal[0]]:
        return None

    denied = ~occ & ~gps
    if denied.sum() < 3:
        return None
    in_zone = _pick(rng, denied, 3)
    rest = ~occ
    for x, y in in_zone + [start, goal]:
        rest[y, x] = False
    beacons = in_zone + _pick(rng, rest, 2)

    return Map(occ=occ, gps=gps, beacons=beacons, start=start, goal=goal)


def make_map(size: int = 200, n_obstacles: int = 15, seed: int = 0,
             n_zones: int = 2, max_attempts: int = 50) -> Map:
    if size < 50:
        raise ValueError("size must be >= 50")
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        m = _try_make(rng, size, n_obstacles, n_zones)
        if m is not None:
            return m
    raise RuntimeError("Could not generate a valid map; try fewer obstacles")


def plot_map(m: Map, path=None, show: bool = True):
    fig = plt.figure(figsize=(7, 7))
    plt.imshow(m.occ, cmap="gray_r", origin="lower")
    zone = np.ma.masked_where(m.gps, ~m.gps)
    plt.imshow(zone, cmap="Reds", alpha=0.3, origin="lower", vmin=0, vmax=1)
    if path:
        plt.plot(*zip(*path), c="royalblue", lw=2, label="Path", zorder=4)
    if m.beacons:
        plt.scatter(*zip(*m.beacons), c="gold", s=60, edgecolors="black",
                    label="Beacons", zorder=5)
    plt.scatter(*m.start, c="limegreen", s=100, edgecolors="black", label="Start", zorder=5)
    plt.scatter(*m.goal, c="red", marker="X", s=120, edgecolors="black", label="Goal", zorder=5)
    plt.legend(loc="upper right")
    dist = np.hypot(m.goal[0] - m.start[0], m.goal[1] - m.start[1])
    plt.title(f"Distance: {dist:.1f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    my_map = make_map(size=200, n_obstacles=25, seed=42)
    print(f"Start: {my_map.start}, Finish: {my_map.goal}")
    plot_map(my_map)