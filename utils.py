import os
import json
import argparse
import vrplib

from pandas import Series
import numpy as np

import instances.utils as instances_utils
from methods.or_tools import ORtools
from methods.own import Own


def parse_experiment():
    # To handle dictionary input
    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split("=")
                getattr(namespace, self.dest)[key] = value

    """Parse arguments for an experiment run"""
    parser = argparse.ArgumentParser(description="Experiment arguments")
    parser.add_argument("--seed", help="Specify random seed")
    parser.add_argument("--problem", help="Specify task. Options: 'CVRP'")
    parser.add_argument("--training", default=None, help="Specify training data")
    parser.add_argument("--method", default="nazari", help="Specify solution method")
    parser.add_argument(
        "--method_settings",
        default={},
        help="Specify method specific parameters",
        nargs="*",
        action=ParseKwargs,
    )
    parser.add_argument(
        "--testing", default=[], help="Specify test sets", type=str, nargs="*"
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Specify device that should be used. GPU: 0 (default), CPU: -1",
    )

    args, unknown = parser.parse_known_args()
    args = vars(args)

    return args


def instance_to_nazari(instance):
    """Convert an instance in the general format to that needed for Nazari"""
    return np.roll(
        np.column_stack((instance["node_coord"], instance["demand"])), -1, axis=0
    ).reshape(-1, instance["dimension"], 3)


def test_cvrp_nazari(model, folder_path):
    """Carry out testing on all examples in a specific folder for a Nazari model"""
    results = {"greedy": {}, "beam": {}}
    routes = {"greedy": {}, "beam": {}}

    for example in next(os.walk(folder_path))[2]:
        if example.endswith("sol"):
            continue
        print(example)
        # Go through all test instances
        data = vrplib.read_instance(f"{folder_path}/{example}")
        if data["edge_weight_type"] != "EUC_2D":
            continue

        if model.args["n_nodes"] != data["dimension"]:
            continue
        model.testing(instance_to_nazari(data), data)
        results["greedy"][example] = model.cost["greedy"]
        results["beam"][example] = model.cost["beam"]
        routes["greedy"][example] = model.routes["greedy"]
        routes["beam"][example] = model.routes["beam"]

    return results, routes


def test_cvrp_algm(method, method_settings, folder_path):
    """Carry out testing on all examples in a specific folder for an algorithm method"""
    results = {}
    routes = {}
    print(folder_path)
    for example in next(os.walk(folder_path))[2]:
        if example.endswith("sol"):
            continue
        print(example)
        # Go through all test instances
        data = vrplib.read_instance(f"{folder_path}/{example}")
        if data["edge_weight_type"] != "EUC_2D":
            continue

        # Set up model
        if method == "ortools":
            model = ORtools(
                data,
                method_settings["init_method"],
                method_settings["improve_method"],
            )
        elif method == "own":
            model = Own(data, method_settings)

        # Get results
        try:
            model.run_all()
        except (AttributeError, SystemError):
            continue

        results[example] = model.cost
        routes[example] = model.routes

    return results, routes


def test_cvrp_rl4co(model, folder_path):
    """Carry out testing on all examples in a specific folder for a RL4CO model"""
    results = {}
    routes = {}

    for example in next(os.walk(folder_path))[2]:
        if example.endswith("sol"):
            continue
        print(example)
        # Go through all test instances
        data = vrplib.read_instance(f"{folder_path}/{example}")
        if data["edge_weight_type"] != "EUC_2D":
            continue

        model.single_test(data)
        results[example] = model.cost
        routes[example] = model.routes

    return results, routes


def test_cvrp_other(model, method, method_settings, test_set):
    """Carry out testing on all examples in a specific folder for any model"""
    results = {}
    routes = {}

    if method == "nazari":
        results, routes = test_cvrp_nazari(model, f"instances/CVRP/{test_set}")
    elif method in ["ortools", "own"]:
        results, routes = test_cvrp_algm(
            method, method_settings, f"instances/CVRP/{test_set}"
        )
    elif method in ["rl4co", "rl4co_tsp"]:
        results, routes = test_cvrp_rl4co(model, f"instances/CVRP/{test_set}")

    return results, routes


def test_cvrp_generate(model, method, method_settings):
    """Carry out testing on all examples across all generate folders for any model"""
    # Running all tests for the generated test instances
    results = {}
    routes = {}
    for subdir in next(os.walk("instances/CVRP/generate"))[1]:
        results[subdir], routes[subdir] = test_cvrp_other(
            model, method, method_settings, f"/generate/{subdir}"
        )

    return results, routes


def test_cvrp(method, method_settings, ident, testing, model=None, save=True):
    """Function for running CVRP testing
    - method - the solution method being applied
    - method_settings - any additional settings for the method
    - ident - the experiment id
    - testing - the testing instances to use
    - model - provided for RL methods, after training"""

    # Setting up dictionaries for saving results
    results_a = {}
    routes_a = {}

    # Running testing
    for test_set in testing:
        print("Starting: ", test_set)

        if test_set == "generate":
            results_b, routes_b = test_cvrp_generate(model, method, method_settings)

        else:
            # Running all tests for the general test instances
            results_a[test_set], routes_a[test_set] = test_cvrp_other(
                model, method, method_settings, test_set
            )

    if save:
        # Saving results
        with open(f"results/exp_{ident}/results_a.json", "w") as f:
            json.dump(results_a, f, indent=2)
        with open(f"results/exp_{ident}/routes_a.json", "w") as f:
            json.dump(routes_a, f, indent=2)
        try:
            with open(f"results/exp_{ident}/results_b.json", "w") as f:
                json.dump(results_b, f, indent=2)
            with open(f"results/exp_{ident}/routes_b.json", "w") as f:
                json.dump(routes_b, f, indent=2)
        except NameError:
            pass
    else:
        results = {"a": results_a}
        routes = {"a": routes_a}
        try:
            results["b"] = results_b
            routes["b"] = routes_b
        except NameError:
            pass
        return results, routes


def test_cvrptw(method, method_settings, ident, testing, model=None, save=True):
    """Function for running CVRP testing
    - method - the solution method being applied
    - method_settings - any additional settings for the method
    - ident - the experiment id
    - testing - the testing instances to use, indicated by numbers for the customer size
    - model - provided for RL methods, after training"""
    # Set up for saving results
    results = {}
    routes = {}
    # Set up for different size versions of instances
    for tester in testing:
        results[tester] = {}
        routes[tester] = {}
    # Running testing
    for example in next(os.walk(f"instances/CVRPTW/Solomon"))[2]:
        # Go through all test instances
        if example.endswith("sol"):
            continue
        print(example)
        data = vrplib.read_instance(
            f"instances/CVRPTW/Solomon/{example}", instance_format="solomon"
        )
        data["type"] = "CVRPTW"
        data["dimension"] = len(data["demand"])
        for tester in testing:
            # Rescale instances as needed
            data2 = instances_utils.shrink_twinstance(data, tester)
            if method == "ortools":
                model = ORtools(
                    data2,
                    method_settings["init_method"],
                    method_settings["improve_method"],
                )
                print("model done")
                try:
                    model.run_all()
                    print("run all")
                    results[tester][example] = model.cost
                    routes[tester][example] = model.routes
                except AttributeError:
                    continue
                except SystemError:
                    try:
                        model.no_vehicles = 2 * model.no_vehicles
                        print("run all")
                        results[tester][example] = model.cost
                        routes[tester][example] = model.routes
                    except SystemError:
                        continue
            elif method == "rl4co":
                model.single_test(data2)
                results[tester][example] = model.cost
                routes[tester][example] = model.routes
    if save:
        # Saving results
        with open(f"results/exp_{ident}/results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(f"results/exp_{ident}/routes.json", "w") as f:
            json.dump(routes, f, indent=2)
    else:
        return results, routes
