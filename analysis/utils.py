import os
import json
from pathlib import Path
from itertools import pairwise
import vrplib
import pandas as pd

import instances.utils as instance_utils


def add_settings(df):
    """Add setting information to a dataframe"""
    settings_df = pd.read_csv("results/settings.csv")
    output = pd.merge(df, settings_df, on='id', how='left')
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
        with open(f"results/baseline_optima.json", "w") as f:
            json.dump(output, f, indent=2)

    return output


def validate_routes(routes, demand, capacity):
    """Check the validity of a route, given the demands and capacity"""

    # Check that all nodes are visited once
    if len(set([i for j in routes for i in j])) != len(demand) - 1:
        print("Incorrect number of visited nodes")
        return 0

    # Check that capacity is not exceeded
    for route in routes:
        if sum(demand[route]) > capacity:
            print("Exceeds capacity")
            return 0

    return 1


def validate_experiment(ident):
    """Carry out validation checks on all of the results for an experiment"""
    options = ["a", "b"]
    paths = {"a": "instances/CVRP", "b": "instances/CVRP/generate"}
    for option in options:
        # Run separately for each of a and b
        output = {}
        with open(f"results/exp_{ident}/routes_{option}.json") as json_data:
            routes = json.load(json_data)
        for test_set in routes:
            output[test_set] = {}
            if "greedy" in routes[test_set]:
                # Procedure for Nazari results
                splits = ["greedy", "beam"]
                for split in splits:
                    if len(routes[test_set][split]) > 0:
                        # Check there are results, and then run through them
                        output[test_set][split] = {}
                        for instance in routes[test_set][split]:
                            # Load instance and run validity check
                            data = vrplib.read_instance(
                                f"{paths[option]}/{test_set}/{instance}"
                            )
                            output[test_set][split][instance] = validate_routes(
                                routes[test_set][split][instance],
                                data["demand"],
                                data["capacity"],
                            )
        # Save results
        with open(f"results/exp_{ident}/validity_{option}.json", "w") as f:
            json.dump(output, f, indent=2)
