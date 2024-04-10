import instances.utils as instances_utils

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

data = instances_utils.import_instance('instances/CVRP/A', 'A-n80-k10')

#Adapted from https://alns.readthedocs.io/en/stable/examples/capacitated_vehicle_routing_problem.html
def plot_solution(data, name="CVRP solution"):
    """
    Plot the routes of the passed-in solution.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(data['solution']['routes'])))

    for idx, route in enumerate(data['solution']['routes']):
        ax.plot(
            [data['instance']["node_coord"][loc][0] for loc in [0] + route + [0]],
            [data['instance']["node_coord"][loc][1] for loc in [0] + route + [0]],
            color=cmap[idx],
            marker='.'
        )

    # Plot the depot
    kwargs = dict(label="Depot", zorder=3, marker="*", s=750)
    ax.scatter(data['instance']["node_coord"][0][0], data['instance']["node_coord"][0][1], c="tab:red", **kwargs)

    ax.set_title(f"{name}\n Total distance: {data['solution']['cost']}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)

plot_solution(data, name="Best known solution")