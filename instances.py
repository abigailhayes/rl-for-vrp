import vrplib
from types import SimpleNamespace
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

data = vrplib.read_instance('instances/A/A-n32-k5.vrp')
bks = SimpleNamespace(**vrplib.read_solution('instances/A/A-n32-k5.sol'))

#From https://alns.readthedocs.io/en/stable/examples/capacitated_vehicle_routing_problem.html
def plot_solution(solution, name="CVRP solution"):
    """
    Plot the routes of the passed-in solution.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(solution.routes)))

    for idx, route in enumerate(solution.routes):
        ax.plot(
            [data["node_coord"][loc][0] for loc in [0] + route + [0]],
            [data["node_coord"][loc][1] for loc in [0] + route + [0]],
            color=cmap[idx],
            marker='.'
        )

    # Plot the depot
    kwargs = dict(label="Depot", zorder=3, marker="*", s=750)
    ax.scatter(*data["node_coord"][0], c="tab:red", **kwargs)

    ax.set_title(f"{name}\n Total distance: {solution.cost}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)

plot_solution(bks, name="Best known solution")