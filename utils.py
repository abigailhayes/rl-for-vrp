import os
import json
import argparse
import vrplib

from pandas import Series
import numpy as np

import instances.utils as instances_utils
from methods.or_tools import ORtools


def parse_experiment():
    # To handle dictionary input
    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split('=')
                getattr(namespace, self.dest)[key] = value

    """Parse arguments for an experiment run"""
    parser = argparse.ArgumentParser(description="Experiment arguments")
    parser.add_argument('--seed', help="Specify random seed")
    parser.add_argument('--problem', help="Specify task. Options: 'CVRP'")
    parser.add_argument('--training', default=None, help="Specify training data")
    parser.add_argument('--method', default='nazari', help="Specify solution method")
    parser.add_argument('--method_settings', default={}, help="Specify method specific parameters", nargs='*', action=ParseKwargs)
    parser.add_argument('--testing', default=[], help="Specify test sets", type=str, nargs="*")
    parser.add_argument('--device', default=0, help="Specify device that should be used. GPU: 0 (default), CPU: -1")

    args, unknown = parser.parse_known_args()
    args = vars(args)

    return args


def instance_to_nazari(instance):
    """Convert an instance in the general format to that needed for Nazari"""
    return np.roll(np.column_stack((instance['node_coord'], instance['demand'])), -1, axis=0).reshape(-1, instance['dimension'], 3)

def test_cvrp(method, method_settings, ident, testing, model=None):
    """ Function for running CVRP testing
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
        print('Starting: ', test_set)

        if test_set == 'generate':
            # Running all tests for the generated test instances
            results_b = {}
            routes_b = {}
            for subdir in next(os.walk('instances/CVRP/generate'))[1]:
                results_b[subdir] = {}
                routes_b[subdir] = {}
                if method == 'nazari':
                    # Nazari has two variants, and these are run adjacently, so need another level of dictionary
                    results_b[subdir]['greedy'] = {}
                    results_b[subdir]['beam'] = {}
                    routes_b[subdir]['greedy'] = {}
                    routes_b[subdir]['beam'] = {}
                for example in next(os.walk(f'instances/CVRP/generate/{subdir}'))[2]:
                    print(example)
                    # Go through all test instances
                    data = vrplib.read_instance(f'instances/CVRP/generate/{subdir}/{example}')
                    if method == 'ortools':
                        model = ORtools(data, method_settings['init_method'],
                                        method_settings['improve_method'])
                        try:
                            model.run_all()
                        except AttributeError:
                            continue
                        results_b[subdir][example] = model.cost
                        routes_b[subdir][example] = model.routes
                    elif method == 'nazari':
                        if model.args['n_nodes'] != data['dimension']:
                            continue
                        model.testing(instance_to_nazari(data), data)
                        results_b[subdir]['greedy'][example] = model.cost['greedy']
                        results_b[subdir]['beam'][example] = model.cost['beam']
                        routes_b[subdir]['greedy'][example] = model.routes['greedy']
                        routes_b[subdir]['beam'][example] = model.routes['beam']

        else:
            # Running all tests for the general test instances
            results_a[test_set] = {}
            routes_a[test_set] = {}
            if method == 'nazari':
                results_a[test_set]['greedy'] = {}
                results_a[test_set]['beam'] = {}
                routes_a[test_set]['greedy'] = {}
                routes_a[test_set]['beam'] = {}
            for example in next(os.walk(f'instances/CVRP/{test_set}'))[2]:
                print(example)
                # Go through all test instances
                if example.endswith('sol'):
                    continue
                data = vrplib.read_instance(f'instances/CVRP/{test_set}/{example}')
                if data['edge_weight_type'] != 'EUC_2D':
                    continue
                if method == 'ortools':
                    model = ORtools(data, method_settings['init_method'], method_settings['improve_method'])
                    try:
                        model.run_all()
                    except AttributeError:
                        continue
                    results_a[test_set][example] = model.cost
                    routes_a[test_set][example] = model.routes
                elif method == 'nazari':
                    if model.args['n_nodes'] != data['dimension']:
                        continue
                    model.testing(instance_to_nazari(data), data)
                    results_a[test_set]['greedy'][example] = model.cost['greedy']
                    results_a[test_set]['beam'][example] = model.cost['beam']
                    routes_a[test_set]['greedy'][example] = model.routes['greedy']
                    routes_a[test_set]['beam'][example] = model.routes['beam']

    # Saving results
    with open(f'results/exp_{ident}/results_a.json', 'w') as f:
        json.dump(results_a, f, indent=2)
    with open(f'results/exp_{ident}/routes_a.json', 'w') as f:
        json.dump(routes_a, f, indent=2)
    try:
        with open(f'results/exp_{ident}/results_b.json', 'w') as f:
            json.dump(results_b, f, indent=2)
        with open(f'results/exp_{ident}/routes_b.json', 'w') as f:
            json.dump(routes_b, f, indent=2)
    except NameError:
        pass
