import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2)

_NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
]


def _octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2) * min(dx, dy)


def astar(blocked: np.ndarray, start: tuple, goal: tuple):
    height, width = blocked.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    def free(x, y):
        return 0 <= x < width and 0 <= y < height and not blocked[y, x]

    if not free(*start) or not free(*goal):
        return None
    if start == goal:
        return [start]

    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    heap = [(_octile(start, goal), start)]

    while heap:
        _, node = heapq.heappop(heap)
        if node in closed:
            continue
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]
        closed.add(node)

        x, y = node
        for dx, dy, cost in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not free(nx, ny):
                continue
            if dx and dy and (blocked[y, nx] or blocked[ny, x]):
                continue
            nb = (nx, ny)
            if nb in closed:
                continue
            new_g = g[node] + cost
            if new_g < g.get(nb, math.inf):
                g[nb] = new_g
                parent[nb] = node
                heapq.heappush(heap, (new_g + _octile(nb, goal), nb))

    return None