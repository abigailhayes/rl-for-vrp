import os
import json
import sys
import time
from datetime import datetime
import numpy as np
import argparse

from statistics import mean
from pandas import Series
import tensorflow as tf

import instances.utils as instances_utils
import methods.cw_savings as cw_savings


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

def apply_method(method, instance):
    """Apply the appropriate method to the example dataset."""
    if method == 'CWSavings':
        output = cw_savings.CWSavings(instance['instance'])
    else:
        raise ValueError("Unrecognised method")

    output.add_sol(instance['solution'])
    output.run_all()
    return output


def get_dir(task):
    """Specifies the directory to run through, based on the task."""
    if task == 'CVRP':
        return './instances/CVRP'
    else:
        raise ValueError("Unrecognised task")


def nested_dict_values(d):
    for v1 in d.values():
        for v2 in v1.values():
            yield v2


def avg_perf(task, method, small=True):
    """Function to run over all available instances and get the average percentage that the algorithm
    is worse by
    Specify:
    - task; CVRP or other
    - method; the algorithm being tested"""
    directory = get_dir(task)
    perc_results = {}
    perc_averages = {}
    results = {}
    averages = {}
    for subdir in next(os.walk(directory))[1]:
        print('Running:', subdir)
        if small & (subdir in ["XML100", "XXL", "X", "Li"]):
            continue
        results[subdir] = {}
        perc_results[subdir] = {}
        for example in [example[:-4] for example in next(os.walk(f'{directory}/{subdir}'))[2] if
                        example.endswith('vrp')]:
            instance = instances_utils.import_instance(f'{directory}/{subdir}', example)
            run = apply_method(method, instance)
            results[subdir][example] = run.cost
            perc_results[subdir][example] = run.perc
        averages[subdir] = Series([*results[subdir].values()]).mean()
        perc_averages[subdir] = Series([*perc_results[subdir].values()]).mean()
    perc_averages['all'] = mean(nested_dict_values(perc_results))

    # Save all outputs in files
    os.makedirs(f'results/{task}/{method}', exist_ok=True)
    with open(f'results/{task}/{method}/perc_results.json', 'w') as f:
        json.dump(perc_results, f, indent=2)
    with open(f'results/{task}/{method}/perc_averages.json', 'w') as f:
        json.dump(perc_averages, f, indent=2)
    with open(f'results/{task}/{method}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(f'results/{task}/{method}/averages.json', 'w') as f:
        json.dump(averages, f, indent=2)



