import os
import json
import argparse
import vrplib

from pandas import Series

import instances.utils as instances_utils
from methods.or_tools import ORtools


def parse_experiment():
    """Parse arguments for an experiment run"""
    parser = argparse.ArgumentParser(description="Experiment arguments")
    parser.add_argument('--seed', help="Specify random seed")
    parser.add_argument('--task', help="Specify task. Options: 'CVRP'")
    parser.add_argument('--training', default=None, help="Specify training data")
    parser.add_argument('--method', default='nazari', help="Specify solution method")
    parser.add_argument('--method_settings', default=None, help="Specify method specific parameters")
    parser.add_argument('--testing', default=None, help="Specify test sets")
    parser.add_argument('--device', default=0, help="Specify device that should be used. GPU: 0 (default), CPU: -1")

    args, unknown = parser.parse_known_args()
    args = vars(args)

    return args


def test_cvrp(method, method_settings, ident, testing, model=None):
    """ Function for running CVRP testing
    - method - the solution method being applied
    - method_settings - any additional settings for the method
    - ident - the experiment id
    - testing - the testing instances to use
    - model - provided for RL methods, after training"""

    # Setting up dictionaries for saving results
    results_a = {}
    averages_a = {}

    # Running testing
    for test_set in testing:

        if test_set == 'generate':
            # Running all tests for the generated test instances
            results_b = {}
            averages_b = {}
            for subdir in next(os.walk('instances/CVRP/generate'))[1]:
                results_b[subdir] = {}
                if method == 'ortools':
                    for example in next(os.walk(f'instances/CVRP/generate/{subdir}'))[2]:
                        data = vrplib.read_instance(f'instances/CVRP/generate/{subdir}/{example}')
                        model = ORtools(data['instance'], method_settings['init_method'],
                                        method_settings['improve_method'])
                        model.run_all()
                        results_b[subdir][example] = model.cost
                    averages_b[subdir] = Series([*results_b[subdir].values()]).mean()

        else:
            # Running all tests for the general test instances
            results_a[test_set] = {}
            if method == 'ortools':
                for example in [example[:-4] for example in next(os.walk(f'instances/CVRP/{test_set}'))[2] if
                                example.endswith('vrp')]:
                    data = instances_utils.import_instance(f'instances/CVRP/{test_set}', example)
                    model = ORtools(data['instance'], method_settings['init_method'], method_settings['improve_method'])
                    model.add_sol(data['solution'])
                    model.run_all()
                    results_a[test_set][example] = model.cost
                averages_a[test_set] = Series([*results_a[test_set].values()]).mean()

    # Saving results
    with open(f'results/exp_{ident}/results_a.json', 'w') as f:
        json.dump(results_a, f, indent=2)
    with open(f'results/exp_{ident}/averages_a.json', 'w') as f:
        json.dump(averages_a, f, indent=2)
    try:
        with open(f'results/exp_{ident}/results_b.json', 'w') as f:
            json.dump(results_b, f, indent=2)
        with open(f'results/exp_{ident}/averages_b.json', 'w') as f:
            json.dump(averages_b, f, indent=2)
    except NameError:
        pass
