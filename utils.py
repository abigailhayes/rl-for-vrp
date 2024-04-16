import os
import json
import sys
import time
from datetime import datetime
import numpy as np

from statistics import mean
from pandas import Series
import tensorflow as tf

import instances.utils as instances_utils
import methods.cw_savings as cw_savings


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


# Shared misc_utils
print_grad = True


class printOut(object):
    def __init__(self,f=None ,stdout_print=True):
        '''
        This class is used for controlling the printing. It will write in a
        file f and screen simultanously.
        '''
        self.out_file = f
        self.stdout_print = stdout_print

    def print_out(self, s, new_line=True):
        """Similar to print but with support to flush and output to a file."""
        if isinstance(s, bytes):
            s = s.decode("utf-8")

        if self.out_file:
            self.out_file.write(s)
            if new_line:
                self.out_file.write("\n")
        self.out_file.flush()

        # stdout
        if self.stdout_print:
            print(s, end="", file=sys.stdout)
            if new_line:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def print_time(self, s, start_time):
        """Take a start time, print elapsed duration, and return a new time."""
        self.print_out("%s, time %ds, %s." % (s, (time.time() - start_time) +"  " +str(time.ctime()) ))
        return time.time()
def get_time():
    """returns formatted current time"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.compat.v1.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True
    return config_proto


def debug_tensor(s, msg=None, summarize=10):
    """Print the shape and value of a tensor at test time. Return a new tensor."""
    if not msg:
        msg = s.name
    return tf.compat.v1.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)


def has_nan(datum, tensor):
    if hasattr(tensor, 'dtype'):
        if (np.issubdtype(tensor.dtype, float) or
                np.issubdtype(tensor.dtype, complex) or
                np.issubdtype(tensor.dtype, np.integer)):
            return np.any(np.isnan(tensor))
        else:
            return False
    else:
        return False


def openAI_entropy(logits):
    # Entropy proposed by OpenAI in their A2C baseline
    a0 = logits - tf.reduce_max(logits, 2, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 2, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_mean(tf.reduce_sum(p0 * (tf.compat.v1.log(z0) - a0), 2))


def softmax_entropy(p0):
    # Normal information theory entropy by Shannon
    return - tf.reduce_sum(p0 * tf.compat.v1.log(p0 + 1e-6), axis=1)


def Dist_mat(A):
    # A is of shape [batch_size x nnodes x 2].
    # return: a distance matrix with shape [batch_size x nnodes x nnodes]
    nnodes = tf.shape(A)[1]
    A1 = tf.tile(tf.expand_dims(A, 1), [1, nnodes, 1, 1])
    A2 = tf.tile(tf.expand_dims(A, 2), [1, 1, nnodes, 1])
    dist = tf.norm(A1 - A2, axis=3)
    return dist
