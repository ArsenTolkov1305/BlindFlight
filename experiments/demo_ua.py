import sys

import matplotlib.pyplot as plt

from uahybrid.maps import make_map, plot_map
from uahybrid.metrics import path_length
from uahybrid.planners.astar import astar
from uahybrid.planners.ua_astar import ua_astar, uncertainty_trace

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
m = make_map(size=200, n_obstacles=25, seed=seed)
plain = astar(m.occ, m.start, m.goal)
ua = ua_astar(m)
if ua is None:
    raise SystemExit("UA-A*: безпечного шляху немає")
trace = uncertainty_trace(m, ua)
print(f"A*: {path_length(plain):.1f}, UA-A*: {path_length(ua):.1f}, max R на шляху: {max(trace):.1f}")

plot_map(m, show=False)
plt.plot(*zip(*plain), c="gray", ls="--", lw=1.5, label="A*")
plt.plot(*zip(*ua), c="royalblue", lw=2, label="UA-A*")
for i in range(0, len(ua), 12):
    plt.gca().add_patch(plt.Circle(ua[i], trace[i], fill=False, ec="purple", lw=0.8, alpha=0.7))
plt.legend(loc="upper right")
plt.show()