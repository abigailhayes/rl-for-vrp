import numpy as np
import tensorflow as tf
import vrplib
import os
import time
from itertools import pairwise

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

    def __init__(self, ident, task):
        self.ident = ident
        tf.compat.v1.reset_default_graph()
        self.args, self.prt = ParseParams(self.ident, task)
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

    def testing(self, test_data=None, instance=None):
        self.agent.inference('testing', test_data)
        # Getting routes in normal format
        self.routing(instance)
        self.get_cost(instance)

    def routing(self, instance):
        self.routes = {}
        self.routes['greedy'] = []
        current = []
        self.test = {}
        self.test['coords']=instance['node_coord']
        for node in [np.where(np.all(np.isin(np.roll(self.agent.test_coords['input_greedy'], 1, axis=0), i), axis=1))[0][0] for i in
                             self.agent.test_coords['coords_greedy']]:
            if node == 0:
                self.routes['greedy'].append(current)
                current = []
            else:
                current.append(int(node))
        self.routes['greedy'].append(current)
        self.routes['greedy'] = [route for route in self.routes['greedy'] if len(route) != 0]

        self.routes['beam'] = []
        current = []
        for node in [np.where(np.all(np.isin(np.roll(self.agent.test_coords['input_beam'], 1, axis=0), i), axis=1))[0][0] for i in
                           self.agent.test_coords['coords_beam']]:
            if node == 0:
                self.routes['beam'].append(current)
                current = []
            else:
                current.append(int(node))
        self.routes['beam'].append(current)
        self.routes['beam'] = [route for route in self.routes['beam'] if len(route) != 0]

    def _get_cost(self, routes, instance):
        """Calculate the total cost of a solution to an instance"""
        costs = 0
        for r in routes:
            for i, j in pairwise([0]+r+[0]):
                costs += instance['edge_weight'][i][j]
        return costs

    def get_cost(self, instance):
        """Calculate the total cost of the current solution to an instance"""
        self.cost = {}
        self.cost['greedy'] = self._get_cost(self.routes['greedy'], instance)
        self.cost['beam'] = self._get_cost(self.routes['beam'], instance)




