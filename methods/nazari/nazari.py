import numpy as np
import tensorflow as tf
import vrplib
import os

from methods.nazari.configs import ParseParams
from methods.nazari.attention_agent import RLAgent
from methods.nazari.utils import Env, reward_func
from methods.nazari.attention import AttentionVRPActor, AttentionVRPCritic

# Data formatting
directory = 'instances/CVRP/generate/random_random-20-normal-normal-1'
working = []
for file in next(os.walk(directory))[2]:
    if os.path.splitext(file)[-1].lower() == '.vrp':
        instance = vrplib.read_instance(f'{directory}/{file}')
        result = np.column_stack((instance['node_coord'], instance['demand']))
        result = np.roll(result, -1, axis=0) # Move the depot to the end
        working.append(result)
output = np.stack(working, axis=0)

# Training
args, prt = ParseParams()
tf.compat.v1.disable_eager_execution()
env = Env(args)
agent = RLAgent(args,
                prt,
                env,
                output,
                reward_func,
                AttentionVRPActor,
                AttentionVRPCritic,
                is_train=args['is_train'])
