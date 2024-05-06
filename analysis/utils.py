import os
from pathlib import Path

import vrplib

import instances.utils as instance_utils

test_sets = ['A', 'B', 'E', 'F', 'M', 'P', 'CMT']
for test_set in test_sets:
    for instance in next(os.walk(f'instances/CVRP/{test_set}'))[2]:
        if instance.endswith('sol'):
            continue
        data = instance_utils.import_instance(f'instances/CVRP/{test_set}', Path(instance).stem)