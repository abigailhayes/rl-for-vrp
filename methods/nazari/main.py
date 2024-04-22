import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import time

from configs import ParseParams

from decode_step import RNNDecodeStep
from attention_agent import RLAgent

from utils import DataGenerator, Env, reward_func
from attention import AttentionVRPActor, AttentionVRPCritic


def main(args, prt):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    dataGen = DataGenerator(args)
    dataGen.reset()
    env = Env(args)
    # create an RL agent
    agent = RLAgent(args,
                    prt,
                    env,
                    dataGen,
                    reward_func,
                    AttentionVRPActor,
                    AttentionVRPCritic,
                    is_train=args['is_train'])
    agent.Initialize(sess)

    # train or evaluate
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


if __name__ == "__main__":
    args, prt = ParseParams()
    # Random
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        prt.print_out("# Set random seed to %d" % random_seed)
        tf.random.set_seed(random_seed)
    tf.compat.v1.reset_default_graph()

    main(args, prt)
