import os
import json
from pathlib import Path
from itertools import pairwise
import vrplib
import pandas as pd

import instances.utils as instance_utils


def add_settings(df):
    """Add setting information to a dataframe"""
    settings_df = pd.read_csv("results/other/settings.csv")
    output = pd.merge(df, settings_df, on="id", how="left")
    return output


def check_instances(table, col_name: str):
    """Boolean indication of whether the full instance set is evaluated, returned as float"""
    output = table[col_name] == max(table[col_name])
    return output * 1.0


def baseline_optima(save=False):
    """Calculate the optima for all the example solutions, using the provided route. This differs slightly from the
    distance provided in the solution."""
    test_sets = ["A", "B", "E", "F", "M", "P", "CMT"]
    output = {}
    for test_set in test_sets:
        output[test_set] = {}
        for instance in next(os.walk(f"instances/CVRP/{test_set}"))[2]:
            # Walk through all instances
            if instance.endswith("sol"):
                continue
            data = instance_utils.import_instance(
                f"instances/CVRP/{test_set}", Path(instance).stem
            )
            # Calculate the route distance
            costs = 0
            for r in data["solution"]["routes"]:
                for i, j in pairwise([0] + r + [0]):
                    costs += data["instance"]["edge_weight"][i][j]
            output[test_set][instance] = costs

    if save:
        # Save output
        with open(f"results/other/baseline_optima.json", "w") as f:
            json.dump(output, f, indent=2)

    return output


def validate_routes(routes, demand, capacity):
    """Check the validity of a route, given the demands and capacity"""

    # Check that all nodes are visited once
    if len(set([i for j in routes for i in j])) != len(demand) - 1:
        return 0

    # Check that capacity is not exceeded
    for route in routes:
        if sum(demand[route]) > capacity:
            return 0

    return 1


def validate_dict(route_dict, test_set):
    """Check the validity of a route dictionary for a particular folder of instances"""
    output = 0
    if len(test_set) > 5:
        # Catching when the instances are from generate
        test_set = f"generate/{test_set}"
    for key in route_dict:
        routes = route_dict[key]
        data = vrplib.read_instance(f"instances/CVRP/{test_set}/{key}")
        output += validate_routes(routes, data["demand"], data["capacity"])
    return output


def average_distance(folder_dict: dict):
    """Given a dictionary with entries for each instance, takes the mean of the values"""
    output = []
    for key in folder_dict:
        output.append(folder_dict[key])
    return sum(output) / len(output)
