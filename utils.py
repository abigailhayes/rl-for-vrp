import os
import json

from statistics import mean
from pandas import Series

import instances.utils as instances_utils
import methods.cw_savings as cw_savings

def get_dir(task):
    """Specifies the directory to run through, based on the task."""
    if task=='CVRP':
        return './instances/CVRP'
    else:
        raise ValueError("Unrecognised task")

def apply_method(method, instance):
    """Apply the appropriate method to the example dataset."""
    if method=='CWSavings':
        output = cw_savings.CWSavings(instance['instance'])
    else:
        raise ValueError("Unrecognised method")

    output.add_sol(instance['solution'])
    output.run_all()
    return output

def NestedDictValues(d):
  for v1 in d.values():
    for v2 in v1.values():
        yield v2

def avg_perf(task, method):
    """Function to run over all available instances and get the average percentage that the algorithm
    is worse by
    Specify:
    - task; CVRP or other
    - method; the algorithm being tested"""
    directory = get_dir(task)
    results = {}
    averages = {}
    for subdir in next(os.walk(directory))[1]:
        results[subdir] = {}
        for example in [example[:-4] for example in next(os.walk(f'{directory}/{subdir}'))[2] if example.endswith('vrp')]:
            instance = instances_utils.import_instance(f'{directory}/{subdir}', example)
            run = apply_method(method, instance)
            results[subdir][example] = run.perc
        averages[subdir] = Series([*results[subdir].values()]).mean()
    averages['all'] = mean(NestedDictValues(results))

    # Save all results in a file
    os.makedirs(f'results/{task}', exist_ok=True)
    os.makedirs(f'results/{task}/{method}', exist_ok=True)
    with open(f'results/{task}/{method}/all_instances.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save averages
    with open(f'results/{task}/{method}/averages.json', 'w') as f:
        json.dump(averages, f, indent=2)

