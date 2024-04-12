import instances.utils as instances_utils

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import vrplib

#data = instances_utils.import_instance('instances/CVRP/A', 'A-n80-k10')
data = vrplib.read_instance(f'instances/clusters-random-50-100-15-1.vrp')


def plot_solution(instance, solution, name="CVRP solution"):
    """
    Plot the routes of the passed-in solution.
    Adapted from https://alns.readthedocs.io/en/stable/examples/capacitated_vehicle_routing_problem.html
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(solution['routes'])))

    for idx, route in enumerate(solution['routes']):
        ax.plot(
            [instance["node_coord"][loc][0] for loc in [0] + route + [0]],
            [instance["node_coord"][loc][1] for loc in [0] + route + [0]],
            color=cmap[idx],
            marker='.'
        )

    # Plot the depot
    kwargs = dict(label="Depot", zorder=3, marker="*", s=750)
    ax.scatter(instance["node_coord"][0][0], instance["node_coord"][0][1], c="tab:red", **kwargs)

    ax.set_title(f"{name}\n Total distance: {solution['cost']}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)


def plot_instance(instance, name="CVRP instance"):
    """
    Plot the nodes of the passed-in instance.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, 1))

    ax.scatter(
        [instance["node_coord"][loc][0] for loc in range(1, instance['dimension'])],
        [instance["node_coord"][loc][1] for loc in range(1, instance['dimension'])],
        color=cmap[0],
        marker='.'
    )

    # Plot the depot
    kwargs = dict(label="Depot", zorder=3, marker="*", s=750)
    ax.scatter(instance["node_coord"][0][0], instance["node_coord"][0][1], c="tab:red", **kwargs)

    ax.set_title(f"{name}\n Customers: {instance['dimension']-1}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)


plot_solution(data['instance'], data['solution'], name="Best known solution")
plot_instance(data)
