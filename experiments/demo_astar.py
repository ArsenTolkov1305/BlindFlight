from uahybrid.maps import make_map, plot_map
from uahybrid.metrics import min_clearance, path_length
from uahybrid.planners.astar import astar

m = make_map(size=200, n_obstacles=25, seed=42)
path = astar(m.occ, m.start, m.goal)
print(f"Start {m.start} -> Goal {m.goal}")
print(f"Path points: {len(path)}, length: {path_length(path):.1f}")
print(f"Min clearance to obstacles: {min_clearance(path, m.occ):.1f} cells")
plot_map(m, path)