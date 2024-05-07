import os
import json
from pathlib import Path
from itertools import pairwise

import instances.utils as instance_utils


def baseline_optima(save = False):
    """Calculate the optima for all the example solutions, using the provided route. This differs slightly from the
    distance provided in the solution."""
    test_sets = ['A', 'B', 'E', 'F', 'M', 'P', 'CMT']
    output = {}
    for test_set in test_sets:
        output[test_set] = {}
        for instance in next(os.walk(f'instances/CVRP/{test_set}'))[2]:
            # Walk through all instances
            if instance.endswith('sol'):
                continue
            data = instance_utils.import_instance(f'instances/CVRP/{test_set}', Path(instance).stem)
            # Calculate the route distance
            costs = 0
            for r in data['solution']['routes']:
                for i, j in pairwise([0] + r + [0]):
                    costs += data['instance']['edge_weight'][i][j]
            output[test_set][instance] = costs

    if save:
        # Save output
        with open(f'results/baseline_optima.json', 'w') as f:
            json.dump(output, f, indent=2)

    return output

def validate_routes(routes, demand, capacity):
    """Check the validity of a route, given the demands and capacity"""

    # Check that all nodes are visited once
    if set([i for j in routes for i in j]) != len(demand)-1:
        print('Incorrect number of visited nodes')
        return 0

    # Check that capacity is not exceeded
    for route in routes:
        if sum(demand[route]) > capacity:
            print('Exceeds capacity')
            return 0

    return 1
