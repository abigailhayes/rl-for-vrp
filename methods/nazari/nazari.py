import numpy as np
import tensorflow as tf
import vrplib
import os
import time

from methods.nazari.configs import ParseParams
from methods.nazari.attention_agent import RLAgent
from methods.nazari.utils import DataGenerator, Env, reward_func
from methods.nazari.attention import AttentionVRPActor, AttentionVRPCritic

tf.compat.v1.reset_default_graph()


# Test data formatting
def data_to_nazari(directory):
    """Function to convert a folder of instances to the format needed for the Nazari setup"""
    working = []
    for file in next(os.walk(directory))[2]:
        if os.path.splitext(file)[-1].lower() == '.vrp':
            instance = vrplib.read_instance(f'{directory}/{file}')
            result = np.column_stack((instance['node_coord'], instance['demand']))
            result = np.roll(result, -1, axis=0) # Move the depot to the end
            working.append(result)
    output = np.stack(working, axis=0)
    np.savetxt(fname="instances/CVRP/nazari/"+os.path.basename(directory), X=output.reshape(-1, instance['dimension'] * 3))


# Set up agent
args, prt = ParseParams()
args['test_data'] = output
dataGen = DataGenerator(args)
dataGen.reset()
tf.compat.v1.disable_eager_execution()
env = Env(args)
agent = RLAgent(args,
                prt,
                env,
                dataGen,
                reward_func,
                AttentionVRPActor,
                AttentionVRPCritic,
                is_train=args['is_train'])

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
agent.Initialize(sess)

# Train
start_time = time.time()
if args['is_train']:
    prt.print_out('Training started ...')
    train_time_beg = time.time()
    for step in range(args['n_train']):
        summary = agent.run_train_step()
        _, _, actor_loss_val, critic_loss_val, actor_gra_and_var_val, critic_gra_and_var_val, \
            R_val, v_val, logprobs_val, probs_val, actions_val, idxs_val = summary

        if step % args['save_interval'] == 0:
            agent.saver.save(sess, args['model_dir'] + '/model.ckpt', global_step=step)

        if step % args['log_interval'] == 0:
            train_time_end = time.time() - train_time_beg
            prt.print_out('Train Step: {} -- Time: {} -- Train reward: {} -- Value: {}'
                          .format(step, time.strftime("%H:%M:%S", time.gmtime(train_time_end)),
                                  np.mean(R_val), np.mean(v_val)))
            prt.print_out('    actor loss: {} -- critic loss: {}'
                          .format(np.mean(actor_loss_val), np.mean(critic_loss_val)))
            train_time_beg = time.time()
        if step % args['test_interval'] == 0:
            agent.inference(args['infer_type'])

else:  # inference
    prt.print_out('Evaluation started ...')
    agent.inference(args['infer_type'])

prt.print_out('Total time is {}'.format(
    time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
