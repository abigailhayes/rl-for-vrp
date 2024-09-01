import os
import json
from pathlib import Path
from itertools import pairwise
import vrplib
import pandas as pd
from statistics import mean

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


def validate_routes_tw(routes, data):
    # First check basic demand requirements
    if validate_routes(routes, data["demand"], data["capacity"]) == 0:
        return 0
    # Since demand is compliant, now check time windows
    else:
        # Route by route
        for route in routes:
            current_time = 0
            working_route = [0] + route + [0]
            # Move through nodes on route
            for i in range(len(working_route)):
                if i == 0:
                    continue
                # Add on travel time and check if within time window, waiting if needed
                current_time += data["edge_weight"][
                    working_route[i], working_route[i - 1]
                ]
                if current_time > data["time_window"][working_route[i], 1]:
                    return 0
                elif current_time < data["time_window"][working_route[i], 0]:
                    current_time = data["time_window"][working_route[i], 0]
                # Add on service time
                current_time += data["service_time"][working_route[i]]
        # When no time violations
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


def validate_dict_tw(route_dict, size):
    """Check the validity of a route dictionary for a particular folder of instances"""
    output = 0
    for key in route_dict:
        routes = route_dict[key]
        data = instance_utils.shrink_twinstance(vrplib.read_instance(f"instances/CVRPTW/Solomon/{key}", instance_format='solomon'), size)
        output += validate_routes_tw(routes, data)
    return output


def average_distance(folder_dict: dict):
    """Given a dictionary with entries for each instance, takes the mean of the values"""
    output = []
    for key in folder_dict:
        output.append(folder_dict[key])
    return sum(output) / len(output)


def average_distance_multi(dict, keys):
    """Given a dictionary with entries for each instance, takes the mean of the values"""
    output = []
    for key in keys:
        for key2 in dict[key]:
            output.append(dict[key][key2])
    return sum(output) / len(output)


def average_distance_tw(subdict, variant):
    temp_dict = {k: v for k, v in subdict.items() if k.startswith(variant)}
    output = average_distance(temp_dict)
    return output


def best_or_means(experiment, instance_count):
    # Load in relevant best OR tools results
    json_path = f"results/other/or_results_{experiment}.json"
    # When data is stored directly for each instance
    output = {}
    if experiment == "c":
        try:
            with open(json_path) as json_data:
                data = json.load(json_data)
            for key in data:
                for variant in ["RC1", "RC2", "R1", "R2", "C1", "C2"]:
                    new_key = variant + "_" + str(key)
                    output[new_key] = sum(
                        [
                            1
                            for instance in data[key].keys()
                            if (
                                instance.startswith(variant)
                                and len(data[key][instance]) > 0
                            )
                        ]
                    )
        except ValueError:
            pass
    else:
        instance_count = instance_count.drop(index=0, axis=0)
        try:
            with open(json_path) as json_data:
                data = json.load(json_data)
            for key in data:
                output[key] = len(data[key])
        except ValueError:
            pass

    # Check if there are enough solutions
    verify = {}
    for key in output:
        if output[key] == max(instance_count[key]):
            verify[key] = 1
        else:
            verify[key] = 0

    avgs = {"id": 0, "notes": "OR tools best"}
    if experiment == "c":
        for key in data:
            for variant in ["RC1", "RC2", "R1", "R2", "C1", "C2"]:
                new_key = variant + "_" + str(key)
                if verify[new_key] == 1:
                    avgs[new_key] = average_distance_tw(
                        {k: v["value"] for k, v in data[key].items() if len(v) > 0},
                        variant,
                    )
                else:
                    avgs[new_key] = 0
    elif experiment == "b":
        for key in data:
            if verify[key] == 1:
                avgs[key] = average_distance(
                    {k: v["value"] for k, v in data[key].items() if len(v) > 0}
                )
            else:
                avgs[key] = 0
    else:
        with open("instances/expt_a_solns.json") as json_data:
            optima = json.load(json_data)
        for key in data:
            if verify[key] == 1:
                if key == "CMT":
                    compare_dict = {
                        inner_key: data[key][inner_key]["value"]
                        for inner_key in [
                            "CMT1.vrp",
                            "CMT2.vrp",
                            "CMT3.vrp",
                            "CMT4.vrp",
                            "CMT5.vrp",
                            "CMT11.vrp",
                            "CMT12.vrp",
                        ]
                    }
                else:
                    compare_dict = {k: v["value"] for k, v in data[key].items()}
                working = []
                for new_key in compare_dict:
                    working.append(
                        (compare_dict[new_key] - optima[key][new_key])
                        / optima[key][new_key]
                    )
                avgs[key] = mean(working)
            else:
                avgs[key] = 0

    return pd.DataFrame.from_dict([avgs])


def best_or_means_group_b(defns):
    json_path = f"results/other/or_results_b.json"
    output = {}
    try:
        with open(json_path) as json_data:
            data = json.load(json_data)
        for key in data:
            output[key] = len(data[key])
    except ValueError:
        pass

    verify = {}
    for key, item in defns.items():
        test = 1
        for col in item:
            if output[col] < 100:
                test = 0
        else:
            verify[key] = test

    avgs = {"id": 0, "notes": "OR tools best"}
    for key, item in defns.items():
        if verify[key] == 1:
            avgs[key] = average_distance_multi(
                {
                    j: {k: v["value"] for k, v in data[j].items() if len(v) > 0}
                    for j in data
                },
                item,
            )

    return pd.DataFrame.from_dict([avgs])
