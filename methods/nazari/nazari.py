import numpy as np
import tensorflow as tf
import vrplib
import os

# Data formatting
directory = 'instances/CVRP/generate/random_random-20-normal-normal-1'
working = []
for file in next(os.walk(directory))[2]:
    if os.path.splitext(file)[-1].lower() == '.vrp':
        print('Running:', file)
        instance = vrplib.read_instance(f'{directory}/{file}')
        result = np.column_stack((instance['node_coord'], instance['demand']))
        result = np.roll(result, -1, axis=0) # Move the depot to the end
        working.append(result)
output = np.stack(working, axis=0)

# Training

