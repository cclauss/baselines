# -*- coding: utf-8 -*-
"""run_atari.py: Train or continue training of ppo2 agent with RUDDER on atari games; Adapted from
baselines.ppo2.run_atari.py

Author -- Michael Widrich
Contact -- widrich@bioinf.jku.at

"""
import os, sys

import datetime as dt
from baselines import bench, logger

import numpy as np

from TeLL.config import Config
from TeLL.utility.plotting import launch_plotting_daemon, save_subplots, save_movie, save_subplots_line_plots
from TeLL.utility.misc import make_sure_path_exists, Tee

# Start subprocess for plotting workers
#  Due to a garbage-collector bug with matplotlib/GPU, launch_plotting_daemon needs so be called before tensorflow
#  import
launch_plotting_daemon(num_workers=3)

if __name__ == "__main__":
    import tensorflow as tf


def train(env_id, num_timesteps, policy, working_dir, config):
    # Original modules
    from baselines.common import set_global_seeds
    from baselines.common.atari_wrappers import make_atari
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    import logging
    import gym
    import os.path as osp
    import tensorflow as tf
    # Module modified for RUDDER
    from baselines.common.vec_env.vec_frame_stack import VecFrameStackNoZeroPadding
    from baselines.common.atari_wrappers import wrap_modified_rr
    from baselines.ppo2_rudder import ppo2_rudder
    from baselines.ppo2_rudder.policies import CnnPolicy, LstmPolicy, LstmPolicyDense
    
    bl_config = config.bl_config
    
    # Set numpy random seed
    rnd_seed = config.get_value('random_seed', 12345)
    rnd_gen = np.random.RandomState(seed=rnd_seed)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get_value("cuda_gpu", "0"))
    
    # Tensorflow configuration
    tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=config.get_value("inter_op_parallelism_threads", 1),
            intra_op_parallelism_threads=config.get_value("intra_op_parallelism_threads", 1),
            log_device_placement=config.get_value("log_device_placement", False)
    )
    tf_config.gpu_options.allow_growth = config.get_value("tf_allow_growth", True)
    
    # Start Tensorflow session
    print("Preparing Logger...")
    
    gym.logger.setLevel(logging.WARN)
    print("Starting session...")
    tf_session = tf.Session(config=tf_config).__enter__()
    
    # Set tensorflow random seed
    tf.set_random_seed(rnd_seed)
    
    # Create parallel environments
    print("Preparing Envionments...", end="")
    
    def make_env(rank):
        def env_fn():
            env = make_atari(env_id)
            env.seed(rnd_seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return wrap_modified_rr(env, episode_life=bl_config['episode_life'],
                                    episode_reward=bl_config['episode_reward'],
                                    episode_frame=bl_config['episode_frame'])
        return env_fn
    
    nenvs = bl_config['num_actors']
    print("creating workers...", end="")
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(rnd_seed)
    print("stacking frames...", end="")
    env = VecFrameStackNoZeroPadding(env, 4)
    print("Done!")
    
    # Enter learning
    policy = {'cnn': CnnPolicy, 'lstmdense': LstmPolicyDense, 'lstm': LstmPolicy}[policy]
    ppo2_rudder.learn(policy=policy, env=env, nsteps=128, nminibatches=4, lam=0.95, gamma=0.99, noptepochs=4,
                      log_interval=1, ent_coef=bl_config['ent_coef'], lr=lambda f: f * 2.5e-4 * bl_config['lr_coef'],
                      cliprange=lambda f: f * 0.1, total_timesteps=int(num_timesteps * 1.1), tf_session=tf_session,
                      working_dir=working_dir, config=config,
                      plotting=dict(save_subplots=save_subplots, save_movie=save_movie,
                                    save_subplots_line_plots=save_subplots_line_plots),
                      rnd_gen=rnd_gen)


def main():
    config = Config()
    working_dir = os.path.join(config.working_dir, config.specs)
    working_dir = os.path.join(working_dir, dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    make_sure_path_exists(working_dir)
    
    with open(os.path.join(working_dir, 'log.txt'), 'a') as logfile:
        sys.stdout = Tee(sys.stdout, logfile, sys.stdout)
        
        bl_config = config.get_value('bl_config')
        
        logger.configure(os.path.join(working_dir, 'baselines'), ['tensorboard', 'log', 'stdout'])
        train(env_id=bl_config['env'], num_timesteps=bl_config['num_timesteps'],
              policy=config.get_value('policy'), working_dir=working_dir, config=config)
        
        sys.stdout.flush()

if __name__ == '__main__':
    main()
