# Import
# from __future__ import print_function

import os
import sys
import time
from datetime import datetime
import warnings
import collections

import tensorflow as tf
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

print_grad = True


# Miscellaneous Utils

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514


class printOut(object):
    def __init__(self, f=None, stdout_print=True):
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
        self.print_out("%s, time %ds, %s." % (s, (time.time() - start_time) + "  " + str(time.ctime())))
        return time.time()

    def print_grad(self, model, last=False):
        # gets a model and prints the second norm of the weights and gradients
        if print_grad:
            for tag, value in model.named_parameters():
                if value.grad is not None:
                    self.print_out('{0: <50}'.format(tag) + "\t-- value:"
                                   + '%.12f' % value.norm().data[0] + "\t -- grad: " + str(value.grad.norm().data[0]))
                else:
                    self.print_out('{0: <50}'.format(tag) + "\t-- value:" +
                                   '%.12f' % value.norm().data[0])
            self.print_out("-----------------------------------")
            if last:
                self.print_out("-----------------------------------")
                self.print_out("-----------------------------------")


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def to_np(x):
    return x.data.cpu().numpy()


def to_vars(x):
    if tf.test.is_built_with_cuda():
        x = x.cuda()
    return x


#  for extracting the gradients
def extract(xVar):
    global yGrad
    yGrad = xVar
    print(yGrad)


def extract_norm(xVar):
    global yGrad
    yGradNorm = xVar.norm()
    print(yGradNorm)


# tensorboard logger
class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                                 height=img.shape[0],
                                                 width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.compat.v1.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def _single_cell(unit_type, num_units, forget_bias, dropout, prt,
                 residual_connection=False, device_str=None):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer

    # Cell Type
    if unit_type == "lstm":
        prt.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
        single_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
            num_units,
            forget_bias=forget_bias)
    elif unit_type == "gru":
        prt.print_out("  GRU", new_line=False)
        single_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))
        prt.print_out("  %s, dropout=%g " % (type(single_cell).__name__, dropout),
                      new_line=False)

    # Residual
    if residual_connection:
        single_cell = tf.compat.v1.nn.rnn_cell.ResidualWrapper(single_cell)
        prt.print_out("  %s" % type(single_cell).__name__, new_line=False)

    # Device Wrapper
    """ if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
        prt.print_out("  %s, device=%s" %
                                        (type(single_cell).__name__, device_str), new_line=False)"""

    return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, prt, num_gpus, base_gpu=0):
    """Create a list of RNN cells."""
    # Multi-GPU
    cell_list = []
    for i in range(num_layers):
        prt.print_out("  cell %d" % i, new_line=False)
        dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0
        single_cell = _single_cell(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            prt=prt,
            residual_connection=(i >= num_layers - num_residual_layers),
            device_str=get_device_str(i + base_gpu, num_gpus),
        )
        prt.print_out("")
        cell_list.append(single_cell)

    return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, prt, num_gpus, base_gpu=0):
    """Create multi-layer RNN cell.

    Args:
        unit_type: string representing the unit type, i.e. "lstm".
        num_units: the depth of each unit.
        num_layers: number of cells.
        num_residual_layers: Number of residual layers from top to bottom. For
            example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
            cells in the returned list will be wrapped with `ResidualWrapper`.
        forget_bias: the initial forget bias of the RNNCell(s).
        dropout: floating point value between 0.0 and 1.0:
            the probability of dropout.  this is ignored if `mode != TRAIN`.
        mode: either tf.contrib.learn.TRAIN/EVAL/INFER
        num_gpus: The number of gpus to use when performing round-robin
            placement of layers.
        base_gpu: The gpu device id to use for the first RNN cell in the
            returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
            as its device id.

    Returns:
        An `RNNCell` instance.
    """

    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           num_residual_layers=num_residual_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode,
                           prt=prt,
                           num_gpus=num_gpus,
                           base_gpu=base_gpu)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_list)


def gradient_clip(gradients, params, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
        tf.summary.scalar("clipped_gradient", tf.linalg.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary


def create_or_load_model(model, model_dir, session, out_dir, name, prt):
    """Create translation model and initialize or load parameters in session."""
    start_time = time.time()
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model.saver.restore(session, latest_ckpt)
        prt.print_out(
            "  loaded %s model parameters from %s, time %.2fs" %
            (name, latest_ckpt, time.time() - start_time))
    else:
        prt.print_out("  created %s model with fresh parameters, time %.2fs." %
                        (name, time.time() - start_time))
        session.run(tf.compat.v1.global_variables_initializer())

    global_step = model.global_step.eval(session=session)
    return model, global_step


def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


def add_summary(summary_writer, global_step, tag, value):
    """Add a new summary to the current summary_writer.
    Useful to log things that are not part of the training graph, e.g., tag=BLEU.
    """
    summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.compat.v1.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True
    return config_proto


def check_tensorflow_version():
    if tf.__version__ < "1.2.1":
        raise EnvironmentError("Tensorflow version must >= 1.2.1")


def debug_tensor(s, msg=None, summarize=10):
    """Print the shape and value of a tensor at test time. Return a new tensor."""
    if not msg:
        msg = s.name
    return tf.compat.v1.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)


def tf_print(tensor, transform=None):
    # Insert a custom python operation into the graph that does nothing but print a tensors value
    def print_tensor(x):
        # x is typically a numpy array here, so you could do anything you want with it,
        # but adding a transformation of some kind usually makes the output more digestible
        print(x if transform is None else transform(x))
        return x

    log_op = tf.compat.v1.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    # Return the given tensor
    return res


# VRP Utils

def create_VRP_dataset(
        n_problems,
        n_cust,
        data_dir,
        seed=None,
        data_type='train'):
    """
    This function creates VRP instances and saves them on disk. If a file is already available,
    it will load the file.
    Input:
        n_problems: number of problems to generate.
        n_cust: number of customers in the problem.
        data_dir: the directory to save or load the file.
        seed: random seed for generating the data.
        data_type: the purpose for generating the data. It can be 'train', 'val', or any string.
    output:
        data: a numpy array with shape [n_problems x (n_cust+1) x 3]
        in the last dimension, we have x,y,demand for customers. The last node is for depot and
        it has demand 0.
     """

    # set random number generator
    n_nodes = n_cust + 1
    if seed is None:
        rnd = np.random
    else:
        rnd = np.random.RandomState(seed)

    # build task name and datafiles
    task_name = 'vrp-size-{}-len-{}-{}.txt'.format(n_problems, n_nodes, data_type)
    fname = os.path.join(data_dir, task_name)

    # create/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = np.loadtxt(fname, delimiter=' ')
        data = data.reshape(-1, n_nodes, 3)
    else:
        print('Creating dataset for {}...'.format(task_name))
        # Generate a training set of size n_problems
        x = rnd.uniform(0, 1, size=(n_problems, n_nodes, 2))
        d = rnd.randint(1, 10, [n_problems, n_nodes, 1])
        d[:, -1] = 0  # demand of depot
        data = np.concatenate([x, d], 2)
        np.savetxt(fname, data.reshape(-1, n_nodes * 3))

    return data


class DataGenerator(object):
    def __init__(self,
                 args):

        """
        This class generates VRP problems for training and test
        Inputs:
            args: the parameter dictionary. It should include:
                args['random_seed']: random seed
                args['test_size']: number of problems to test
                args['n_nodes']: number of nodes
                args['n_cust']: number of customers
                args['batch_size']: batchsize for training

        """
        self.args = args
        self.rnd = np.random.RandomState(seed=args['random_seed'])
        print('Created train iterator.')

        # create test data
        self.n_problems = args['test_size']
        # self.test_data = args['test_data']
        self.test_data = create_VRP_dataset(self.n_problems, args['n_cust'], args['data_dir'],
                                            seed=args['random_seed'] + 1, data_type='test')

        self.reset()

    def reset(self):
        self.count = 0

    def get_train_next(self):
        """
        Get next batch of problems for training
        Returns:
            input_data: data with shape [batch_size x max_time x 3]
        """

        input_pnt = self.rnd.uniform(0, 1,
                                     size=(self.args['batch_size'], self.args['n_nodes'], 2))

        demand = self.rnd.randint(1, 10, [self.args['batch_size'], self.args['n_nodes']])
        demand[:, -1] = 0  # demand of depot

        input_data = np.concatenate([input_pnt, np.expand_dims(demand, 2)], 2)

        return input_data

    def get_test_next(self):
        """
        Get next batch of problems for testing
        """
        if self.count < self.args['test_size']:
            input_pnt = self.test_data[self.count:self.count + 1]
            self.count += 1
        else:
            warnings.warn("The test iterator reset.")
            self.count = 0
            input_pnt = self.test_data[self.count:self.count + 1]
            self.count += 1

        return input_pnt

    def get_test_all(self):
        """
        Get all test problems
        """
        return self.test_data


class State(collections.namedtuple("State",
                                   ("load",
                                    "demand",
                                    'd_sat',
                                    "mask"))):
    pass


class Env(object):
    def __init__(self,
                 args):
        """
        This is the environment for VRP.
        Inputs:
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of nodes in VRP
                args['n_custs']: number of customers in VRP
                args['input_dim']: dimension of the problem which is 2
        """
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        self.n_cust = args['n_cust']
        self.input_dim = args['input_dim']
        self.input_data = tf.compat.v1.placeholder(tf.float32, shape=[None, self.n_nodes, self.input_dim])

        self.input_pnt = self.input_data[:, :, :2]
        self.demand = self.input_data[:, :, -1]
        self.batch_size = tf.shape(self.input_pnt)[0]

    def reset(self, beam_width=1):
        """
        Resets the environment. This environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        """

        # dimensions
        self.beam_width = beam_width
        self.batch_beam = self.batch_size * beam_width

        self.input_pnt = self.input_data[:, :, :2]
        self.demand = self.input_data[:, :, -1]

        # modify the self.input_pnt and self.demand for beam search decoder
        #         self.input_pnt = tf.tile(self.input_pnt, [self.beam_width,1,1])

        # demand: [batch_size * beam_width, max_time]
        # demand[i] = demand[i+batchsize]
        self.demand = tf.tile(self.demand, [self.beam_width, 1])

        # load: [batch_size * beam_width]
        self.load = tf.ones([self.batch_beam]) * self.capacity

        # create mask
        self.mask = tf.zeros([self.batch_size * beam_width, self.n_nodes],
                             dtype=tf.float32)

        # update mask -- mask if customer demand is 0 and depot
        self.mask = tf.concat([tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
                               tf.ones([self.batch_beam, 1])], 1)

        state = State(load=self.load,
                      demand=self.demand,
                      d_sat=tf.zeros([self.batch_beam, self.n_nodes]),
                      mask=self.mask)

        return state

    def step(self,
             idx,
             beam_parent=None):
        """
        runs one step of the environment and updates demands, loads and masks
        """

        # if the environment is used in beam search decoder
        if beam_parent is not None:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(self.batch_size), tf.int64),
                                                  [self.beam_width]), 1)
            # batchedBeamIdx:[batch_size*beam_width]
            batchedBeamIdx = batchBeamSeq + tf.cast(self.batch_size, tf.int64) * beam_parent
            # demand:[batch_size*beam_width x sourceL]
            self.demand = tf.gather_nd(self.demand, batchedBeamIdx)
            # load:[batch_size*beam_width]
            self.load = tf.gather_nd(self.load, batchedBeamIdx)
            # MASK:[batch_size*beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask, batchedBeamIdx)

        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_beam), tf.int64), 1)
        batched_idx = tf.concat([BatchSequence, idx], 1)

        # how much the demand is satisfied
        d_sat = tf.minimum(tf.gather_nd(self.demand, batched_idx), self.load)

        # update the demand
        d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(self.demand), tf.int64))
        self.demand = tf.subtract(self.demand, d_scatter)

        # update load
        self.load -= d_sat

        # refill the truck -- idx: [10,9,10] -> load_flag: [1 0 1]
        load_flag = tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32), 1)
        self.load = tf.multiply(self.load, 1 - load_flag) + load_flag * self.capacity

        # mask for customers with zero demand
        self.mask = tf.concat([tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
                               tf.zeros([self.batch_beam, 1])], 1)

        # mask if load= 0
        # mask if in depot and there is still a demand

        self.mask += tf.concat([tf.tile(tf.expand_dims(tf.cast(tf.equal(self.load, 0),
                                                               tf.float32), 1), [1, self.n_cust]),
                                tf.expand_dims(
                                    tf.multiply(tf.cast(tf.greater(tf.reduce_sum(self.demand, 1), 0), tf.float32),
                                                tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32))), 1)], 1)

        state = State(load=self.load,
                      demand=self.demand,
                      d_sat=d_sat,
                      mask=self.mask)

        return state


def reward_func(sample_solution):
    """The reward for the VRP task is defined as the
    negative value of the route length

    Args:
        sample_solution : a list tensor of size decode_len of shape [batch_size x input_dim]
        demands satisfied: a list tensor of size decode_len of shape [batch_size]

    Returns:
        rewards: tensor of size [batch_size]

    Example:
        sample_solution = [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
        sourceL = 3
        batch_size = 2
        input_dim = 2
        sample_solution_tilted[ [[5,5]
                                                    #  [6,6]]
                                                    # [[1,1]
                                                    #  [2,2]]
                                                    # [[3,3]
                                                    #  [4,4]] ]
    """
    # make init_solution of shape [sourceL x batch_size x input_dim]

    # make sample_solution of shape [sourceL x batch_size x input_dim]
    sample_solution = tf.stack(sample_solution, 0)

    sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1], 0),
                                        sample_solution[:-1]), 0)
    # get the reward based on the route lengths

    route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow(
        (sample_solution_tilted - sample_solution), 2), 2), .5), 0)
    return route_lens_decoded


# Shared misc_utils

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
