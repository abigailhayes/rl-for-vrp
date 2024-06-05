import instances.utils as instances_utils

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import vrplib
import json

with open("results/exp_34/routes_b.json") as json_data:
    route_file = json.load(json_data)
with open("results/exp_34/results_b.json") as json_data:
    cost_file = json.load(json_data)

data = {}
data['instance'] = vrplib.read_instance(f'instances/CVRP/generate/cluster_centre-10-30-100-42/clusters-centre-10-100-30-0.vrp')
data['solution'] = {'routes': [route for route in route_file['cluster_centre-10-30-100-42']['clusters-centre-10-100-30-0.vrp'] if len(route)>0],
                    'cost': cost_file['cluster_centre-10-30-100-42']['clusters-centre-10-100-30-0.vrp']}


def plot_solution(instance, solution, name="CVRP solution", demand=False):
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
    kwargs = dict(s=250)
    ax.scatter(instance["node_coord"][0][0], instance["node_coord"][0][1], c="tab:red", **kwargs)

    ax.set_title(f"{name}\n Total distance: {solution['cost']}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")

    if demand:
        for n, [xi, yi] in enumerate(instance['node_coord'][1:]):
            plt.text(xi, yi, instance['demand'][n], va='bottom', ha='center')


def plot_instance(instance, name="CVRP instance", demand=False):
    """
    Plot the nodes of the passed-in instance.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, 1))

    ax.scatter(
        [instance["node_coord"][loc][0] for loc in range(1, instance['dimension'])],
        [instance["node_coord"][loc][1] for loc in range(1, instance['dimension'])],
        color=cmap[0]
    )

    # Plot the depot
    kwargs = dict(s=250)
    ax.scatter(instance["node_coord"][0][0], instance["node_coord"][0][1], c="tab:red", **kwargs)

    ax.set_title(f"{name}\n Customers: {instance['dimension'] - 1}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")

    if demand:
        for n, [xi, yi] in enumerate(instance['node_coord'][1:]):
            plt.text(xi, yi, instance['demand'][n], va='bottom', ha='center')



plot_instance(data)
plot_solution(data['instance'], data['solution'], name="Best known solution")
