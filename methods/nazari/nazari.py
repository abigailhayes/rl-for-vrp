import numpy as np
import tensorflow as tf
import vrplib
import os
import time

from methods.nazari.configs import ParseParams
from methods.nazari.attention_agent import RLAgent
from methods.nazari.utils import DataGenerator, Env, reward_func
from methods.nazari.attention import AttentionVRPActor, AttentionVRPCritic


# Test data formatting
def data_to_nazari(directory):
    """Function to convert a folder of instances to the format needed for the Nazari setup"""
    working = []
    for file in next(os.walk(directory))[2]:
        if os.path.splitext(file)[-1].lower() == '.vrp':
            instance = vrplib.read_instance(f'{directory}/{file}')
            result = np.column_stack((instance['node_coord'], instance['demand']))
            result = np.roll(result, -1, axis=0)  # Move the depot to the end
            working.append(result)
    output = np.stack(working, axis=0)
    np.savetxt(fname="instances/CVRP/nazari/" + os.path.basename(directory),
               X=output.reshape(-1, instance['dimension'] * 3))


class Nazari():

    def __init__(self, ident):
        self.ident = ident
        tf.compat.v1.reset_default_graph()
        self.args, self.prt = ParseParams(self.ident)
        self.dataGen = DataGenerator(self.args)
        self.dataGen.reset()
        tf.compat.v1.disable_eager_execution()
        self.env = Env(self.args)
        self.agent = RLAgent(self.args,
                             self.prt,
                             self.env,
                             self.dataGen,
                             reward_func,
                             AttentionVRPActor,
                             AttentionVRPCritic,
                             is_train=self.args['is_train'])
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=self.config)
        self.agent.Initialize(self.sess)

    def train_model(self):
        start_time = time.time()
        if self.args['is_train']:
            self.prt.print_out('Training started ...')
            train_time_beg = time.time()
            for step in range(self.args['n_train']):
                summary = self.agent.run_train_step()
                _, _, actor_loss_val, critic_loss_val, actor_gra_and_var_val, critic_gra_and_var_val, \
                    R_val, v_val, logprobs_val, probs_val, actions_val, idxs_val = summary

                if step % self.args['save_interval'] == 0:
                    self.agent.saver.save(self.sess, self.args['model_dir'] + '/model.ckpt', global_step=step)

                if step % self.args['log_interval'] == 0:
                    train_time_end = time.time() - train_time_beg
                    self.prt.print_out('Train Step: {} -- Time: {} -- Train reward: {} -- Value: {}'
                                  .format(step, time.strftime("%H:%M:%S", time.gmtime(train_time_end)),
                                          np.mean(R_val), np.mean(v_val)))
                    self.prt.print_out('    actor loss: {} -- critic loss: {}'
                                  .format(np.mean(actor_loss_val), np.mean(critic_loss_val)))
                    train_time_beg = time.time()
                if step % self.args['test_interval'] == 0:
                    self.agent.inference(self.args['infer_type'])

            self.prt.print_out('Total time is {}'.format(
                time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))

    def inference(self, test_data=None):
        start_time = time.time()
        self.prt.print_out('Evaluation started ...')
        self.agent.inference(self.args['infer_type'], test_data)
        self.prt.print_out('Total time is {}'.format(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))

