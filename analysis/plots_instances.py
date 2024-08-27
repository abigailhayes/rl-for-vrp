import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import vrplib
import json
import os
import argparse


def parse_plots():
    """Parse arguments for plot generation
    expt, expt_ids: list, instance_name, instance_set, demand=False"""
    parser = argparse.ArgumentParser(description="Plot arguments")
    parser.add_argument("--expt", help="Experiment - a, b or c")
    parser.add_argument(
        "--expt_ids", default=[], help="Experiment ids to plot", type=str, nargs="*"
    )
    parser.add_argument("--instance_name", help="Instance name including file type")
    parser.add_argument("--instance_set", help="Folder containing the instance")
    parser.add_argument("--demand", default=False, help="Whether demand is on plots")

    args, unknown = parser.parse_known_args()
    args = vars(args)

    return args


def plot_solution(instance, solution, name, experiment_id, demand=False):
    """
    Plot the routes of the passed-in solution.
    Adapted from https://alns.readthedocs.io/en/stable/examples/capacitated_vehicle_routing_problem.html
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, len(solution["routes"])))

    for idx, route in enumerate(solution["routes"]):
        ax.plot(
            [instance["node_coord"][loc][0] for loc in [0] + route + [0]],
            [instance["node_coord"][loc][1] for loc in [0] + route + [0]],
            color=cmap[idx],
            marker=".",
        )

    # Plot the depot
    kwargs = dict(s=250)
    ax.scatter(
        instance["node_coord"][0][0],
        instance["node_coord"][0][1],
        c="tab:red",
        **kwargs,
    )

    ax.set_title(f"{name}: Expt {experiment_id}\n Total distance: {solution['cost']}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")

    if demand:
        for n, [xi, yi] in enumerate(instance["node_coord"][1:]):
            plt.text(xi, yi, instance["demand"][n], va="bottom", ha="center")

    plt.savefig(f"analysis/plots/{name}/expt_{experiment_id}.jpg")


def plot_instance(instance, name, demand=False):
    """
    Plot the nodes of the passed-in instance.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = matplotlib.cm.rainbow(np.linspace(0, 1, 1))

    ax.scatter(
        [instance["node_coord"][loc][0] for loc in range(1, instance["dimension"])],
        [instance["node_coord"][loc][1] for loc in range(1, instance["dimension"])],
        color=cmap[0],
    )

    # Plot the depot
    kwargs = dict(s=250)
    ax.scatter(
        instance["node_coord"][0][0],
        instance["node_coord"][0][1],
        c="tab:red",
        **kwargs,
    )

    ax.set_title(f"{name}\n Customers: {instance['dimension'] - 1}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")

    if demand:
        for n, [xi, yi] in enumerate(instance["node_coord"][1:]):
            plt.text(xi, yi, instance["demand"][n], va="bottom", ha="center")

    plt.savefig(f"analysis/plots/{name}/instance.jpg")


def generate_plots(expt, expt_ids: list, instance_name, instance_set, demand=False):
    """Generates a set of plots for a particular instance
    - expt - the id for the overall experiment e.g. a or b
    - expt_ids - the experiment run solutions to include
    - instance_name - the specific name of the instance
    - instance_set - the set the instance belongs to i.e. the next level folder
    - demand - whether to include the demand of each node in the plot"""

    # Remove .vrp
    short_name = instance_name.replace(".vrp", "")

    # Create an instance specific folder
    if not os.path.exists(f"analysis/plots/{short_name}"):
        os.makedirs(f"analysis/plots/{short_name}")

    # Make sure the folder path is complete
    if expt == "b":
        instance_folder = f"generate/{instance_set}"
    elif expt == "a":
        instance_folder = instance_set

    # Plot the instance without routes, and the optimal solution if available
    instance = vrplib.read_instance(f"instances/CVRP/{instance_folder}/{instance_name}")
    plot_instance(instance, short_name, demand)
    if expt == "a":
        solution = vrplib.read_solution(
            f"instances/CVRP/{instance_folder}/{short_name}.sol"
        )
        plot_solution(instance, solution, short_name, "optimum", demand)

    # A solution plot for each experiment id
    for expt_id in expt_ids:
        try:
            with open(f"results/exp_{expt_id}/routes_{expt}.json") as json_data:
                route_file = json.load(json_data)
            with open(f"results/exp_{expt_id}/results_{expt}.json") as json_data:
                cost_file = json.load(json_data)
            solution = {
                "routes": [
                    route
                    for route in route_file[instance_set][instance_name]
                    if len(route) > 0
                ],
                "cost": cost_file[instance_set][instance_name],
            }
            plot_solution(instance, solution, short_name, expt_id, demand)
        except (OSError, KeyError):
            pass


def main():
    args = parse_plots()
    generate_plots(args['expt'], args['expt_ids'], args['instance_name'], args['instance_set'], args['demand'])


if __name__ == "__main__":
    main()
