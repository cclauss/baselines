# -*- coding: utf-8 -*-
"""ppo2_rudder.py: Adaption of baselines.ppo2.ppo2.py in baselines package for RUDDER for atari games

Author -- Michael Widrich
Contact -- widrich@bioinf.jku.at

"""

import os
import sys
import time
from collections import deque
import joblib
import numpy as np

import tensorflow as tf
from baselines import logger
from baselines.common import explained_variance
from baselines.common.misc_util import relatively_safe_pickle_dump, pickle_load

from baselines.ppo2_rudder.lessons_buffer import LessonReplayBuffer
from baselines.ppo2_rudder.reward_redistribution import RewardRedistributionModel
from TeLL.utility.misc import make_sure_path_exists


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train, nsteps, ent_coef, vf_coef,
                 max_grad_norm, env,
                 reward_redistribution_config, observation_network_config, lstm_network_config, training_config,
                 exploration_config):
        """Model class managing policy network inference and updates, as in baselines.ppo2.ppo2.py, adapted for RUDDER
        
        Parameters
        -------
        tf_session : tensorflow session
            tensorflow session to compute the graph in
        ob_space
            Baselines ob_space object (see ppo2_rudder.py); must provide .shape attribute for (x, y, c) shapes;
        ac_space
            Baselines ac_space object (see ppo2_rudder.py); must provide .n attribute for number of possible actions;
        nbatch_act : int
            Batchsize for stepping through environment
        nbatch_train : int
            Batchsize for training
        nsteps : int
            Fixed number of timesteps to process at once
        ent_coef : float
            Entropy coefficient
        vf_coef : float
            Value function coefficient (weighting of vf vs. redistributed reward)
        max_grad_norm : float
            Clipping value for gradients
        env : any
            Environment class (not used but kept for compatibility reasons)
        reward_redistribution_config : dict
            Dictionary containing config for reward redistribution:
            -----
            lambda_eligibility_trace : float
                Eligibility trace value for redistributed reward
            vf_contrib : float
                Weighting of original value function (vf) vs. redistributed reward (rr), s.t.
                :math:`reward = vf \cdot vf\_contrib + rr \cdot (1-vf\_contrib)`
            use_reward_redistribution_quality_threshold : float
                Quality of reward redistribution has to exceed use_reward_redistribution_quality_threshold to be used;
                use_reward_redistribution_quality_threshold range is [0,1]; Quality measure is the squared prediction
                error, as described in RUDDER paper;
            use_reward_redistribution : bool
                Use reward redistribution?
            rr_junksize : int
                Junksize for reward redistribution; Junks overlap by 1 half each
            cont_pred_w : float
                Weighting of continous prediciton loss vs. prediction loss of final return at last timestep
            intgrd_steps : int
                Stepsize for integrated gradients
            intgrd_batchsize : int
                Integrated gradients is computed batch-wise if intgrd_batchsize > 1
        observation_network_config : dict
            Dictionary containing config for observation network that processes observations and feeds them to LSTM
            network:
            -----
            show_states : bool
                Show frames to network?
            show_statedeltas : bool
                Show frame deltas to network?
            prepoc_states : list of dicts
                Network config to preprocess frames
            prepoc_deltas : list of dicts
                Network config to preprocess frame deltas
            prepoc_observations : list of dicts
                Network config to preprocess features from frame and frame-delta preprocessing networks
        lstm_network_config : dict
            Dictionary containing config for LSTM network:
            -----
            show_actions : bool
                Show taken actions to LSTM?
            reversed : bool
                Process game sequence in reversed order?
            layers : list of dicts
                Network config for LSTM network and optional additional dense layers
            initializations : dict
                Initialization config for LSTM network
            timestep_encoding : dict
                Set "max_value" and "triangle_span" for TeLL.utiltiy.misc_tensorflow.TriangularValueEncoding class
        training_config : dict
            Dictionary containing config for training and update procedure:
            -----
            n_no_rr_updates : int
                Number of updates to perform without training or using reward redistribution network
            n_pretrain_games : int
                Number of games to pretrain the reward redistribution network without using it;
            downscale_lr_policylag : bool
                Downscale learningrate permanently if policy lag gets too large?
            optimizer : tf.train optimizer
                Optimizer in tf.train, e.g. "AdamOptimizer"
            optimizer_params : dict
                Kwargs for optimizer
            l1 : float
                Weighting for l1 weight regularization
            l2 : float
                Weighting for l2 weight regularization
            clip_gradients : float
                Threshold for clipping gradients (clipping by norm)
        exploration_config : dict
            Dictionary containing config for exploration:
            -----
            sample_actions_from_softmax : bool
                True: Apply softmax to policy network output and use it as probabilities to pick an action
                False: Use the max. policy network output as action
            temporal_safe_exploration : bool
                User RUDDER safe exploration
            save_pi_threshold : float
                Threshold value in range [0,1] for safe actions in RUDDER safe exploration
        """
        sess = tf.get_default_session()
        
        #
        # Create policy networks for acting, training, and to get features for reward redistribution model
        # (Variables are reused across networks)
        #
        act_model = policy(sess, ob_space, ac_space, nbatch_act,
                           reward_redistribution_config, observation_network_config, lstm_network_config,
                           training_config, exploration_config, nsteps=1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train,
                             reward_redistribution_config, observation_network_config, lstm_network_config,
                             training_config, exploration_config, nsteps=nsteps, reuse=True)
        simple_model = policy(sess, ob_space, ac_space, 1,
                              reward_redistribution_config, observation_network_config, lstm_network_config,
                              training_config, exploration_config, nsteps=1, reuse=True)
        
        # Safe connection to reward redistribution model to access observation features
        self.rr_model = act_model.rr_observation_model
        self.rr_model_training = train_model.rr_observation_model
        
        #
        # Placeholders
        #
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        RR = tf.placeholder(tf.float32, [None])  # Redistributed reward
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LSTM_RELSQERR = tf.placeholder(tf.float32, [None])  # Relative squared error of LSTM prediction
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        
        #
        # Losses
        #
        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())
        
        # Value function loss for environment reward
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, -CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        
        # Advantage with vf and reward redistribution
        scaled_rr = LSTM_RELSQERR * RR + (1 - LSTM_RELSQERR) * ADV
        ADV = (reward_redistribution_config['vf_contrib'] * ADV
               + (1 - reward_redistribution_config['vf_contrib']) * scaled_rr)
        
        # Policy gradient loss
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        
        # Downscale learning rate based on exponential running mean of approxkl to deal with policy lag
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            g = tf.get_default_graph()
            
            emavg_approxkl = tf.get_variable('emavg_approxkl', initializer=tf.constant(0, dtype=tf.float32))
            lr_scaling = tf.get_variable('lr_scaling', initializer=tf.constant(1, dtype=tf.float32))
            
            if training_config['downscale_lr_policylag']:
                update_emavg_approxkl = tf.assign(emavg_approxkl, emavg_approxkl*0.99 + approxkl*0.01)
                with g.control_dependencies([update_emavg_approxkl]):
                    # Downscaling lr if emavg_approxkl > 0.1
                    should_downscale_lr = tf.greater(emavg_approxkl, 0.1)
                    # Reset emavg_approxkl after downscaling lr
                    reduce_emavg_approxkl = tf.assign(emavg_approxkl, tf.cond(should_downscale_lr,
                                                                              lambda: emavg_approxkl * 0,
                                                                              lambda: emavg_approxkl))
                    
                    with g.control_dependencies([reduce_emavg_approxkl]):
                        # Downscale learning rate by 1% of its original value each time
                        update_lr_scaling = tf.assign(lr_scaling, tf.cond(should_downscale_lr,
                                                                          lambda: lr_scaling - 0.01,
                                                                          lambda: lr_scaling))
                        with g.control_dependencies([update_lr_scaling]):
                            # Stop downscaling learningrate if it reached 1% of its original value
                            lr_scaling = tf.clip_by_value(lr_scaling, clip_value_max=1., clip_value_min=0.01)
        
        # Combine policy gradient, entropy, and value function losses
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        
        #
        # Updates
        #
        with tf.variable_scope('model'):
            params = tf.trainable_variables(scope='model')
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR*lr_scaling, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)
        
        def train(lr, cliprange, obs, returns, rewards, masks, actions, values, neglogpacs, rr_quality, states=None):
            """'rewards' is the redistributed reward mix"""
            # ADV for environment reward
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, RR: rewards, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values, LSTM_RELSQERR: rr_quality}
            
            if states is not None:
                td_map[train_model.S] = states
            
            td_map[train_model.M] = masks
            
            return sess.run([pg_loss, vf_loss, entropy, approxkl, clipfrac, lr_scaling, emavg_approxkl, _train],
                            td_map)[:-1]
        
        self.loss_names = ['policy_loss', 'vf_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'lr_scaling',
                           'emavg_approxkl']
        
        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.simple_model = simple_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.action = act_model.action
        self.emavg_approxkl = emavg_approxkl
        self.lr_scaling = lr_scaling


class Runner(object):
    def __init__(self, *, env, model, nsteps, gamma, lam, rr_model,
                 reward_redistribution_config, observation_network_config, lstm_network_config, training_config,
                 exploration_config, lessons_buffer_config,
                 summary_writer=None, n_pretrain_games=0, rnd_gen=None, verbose=False):
        """Model class managing policy network inference and updates, as in baselines.ppo2.ppo2.py, adapted for RUDDER
        
        Parameters
        -------
        reward_redistribution_config : dict
            Dictionary containing config for reward redistribution:
            -----
            lambda_eligibility_trace : float
                Eligibility trace value for redistributed reward
            vf_contrib : float
                Weighting of original value function (vf) vs. redistributed reward (rr), s.t.
                :math:`reward = vf \cdot vf\_contrib + rr \cdot (1-vf\_contrib)`
            use_reward_redistribution_quality_threshold : float
                Quality of reward redistribution has to exceed use_reward_redistribution_quality_threshold to be used;
                use_reward_redistribution_quality_threshold range is [0,1]; Quality measure is the squared prediction
                error, as described in RUDDER paper;
            use_reward_redistribution : bool
                Use reward redistribution?
            rr_junksize : int
                Junksize for reward redistribution; Junks overlap by 1 half each
            cont_pred_w : float
                Weighting of continous prediciton loss vs. prediction loss of final return at last timestep
            intgrd_steps : int
                Stepsize for integrated gradients
            intgrd_batchsize : int
                Integrated gradients is computed batch-wise if intgrd_batchsize > 1
        observation_network_config : dict
            Dictionary containing config for observation network that processes observations and feeds them to LSTM
            network:
            -----
            show_states : bool
                Show frames to network?
            show_statedeltas : bool
                Show frame deltas to network?
            prepoc_states : list of dicts
                Network config to preprocess frames
            prepoc_deltas : list of dicts
                Network config to preprocess frame deltas
            prepoc_observations : list of dicts
                Network config to preprocess features from frame and frame-delta preprocessing networks
        lstm_network_config : dict
            Dictionary containing config for LSTM network:
            -----
            show_actions : bool
                Show taken actions to LSTM?
            reversed : bool
                Process game sequence in reversed order?
            layers : list of dicts
                Network config for LSTM network and optional additional dense layers
            initializations : dict
                Initialization config for LSTM network
            timestep_encoding : dict
                Set "max_value" and "triangle_span" for TeLL.utiltiy.misc_tensorflow.TriangularValueEncoding class
        training_config : dict
            Dictionary containing config for training and update procedure:
            -----
            n_no_rr_updates : int
                Number of updates to perform without training or using reward redistribution network
            n_pretrain_games : int
                Number of games to pretrain the reward redistribution network without using it;
            downscale_lr_policylag : bool
                Downscale learningrate permanently if policy lag gets too large?
            optimizer : tf.train optimizer
                Optimizer in tf.train, e.g. "AdamOptimizer"
            optimizer_params : dict
                Kwargs for optimizer
            l1 : float
                Weighting for l1 weight regularization
            l2 : float
                Weighting for l2 weight regularization
            clip_gradients : float
                Threshold for clipping gradients (clipping by norm)
        exploration_config : dict
            Dictionary containing config for exploration:
            -----
            sample_actions_from_softmax : bool
                True: Apply softmax to policy network output and use it as probabilities to pick an action
                False: Use the max. policy network output as action
            temporal_safe_exploration : bool
                User RUDDER safe exploration
            save_pi_threshold : float
                Threshold value in range [0,1] for safe actions in RUDDER safe exploration
        lessons_buffer_config : dict
            Dictionary containing config for lessons buffer:
            -----
            type : str
                Type of lessons buffer to use;
                "constant" is the buffer described in RUDDER paper;
                "off" to use not buffer;
            n_replay_updates : int
                Number of updates on lessons buffer per update on a new episode
            buffer_size : int
                Number of lessons to store in lessons buffer;
            traina2c : bool
                Train a2c model on samples from lessons buffer?
        """
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = np.array([False for _ in range(nenv)])
        
        self.nenvs = nenv
        self.sess = tf.get_default_session()
        self.rr_model = rr_model
        
        self.buffered_incomplete_games = [[] for _ in range(self.nenvs)]
        self.buffered_complete_games = [[] for _ in range(self.nenvs)]
        self.buffer_actor_states = model.initial_state
        self.buffer_actor_obs = self.obs.copy()
        self.buffer_actor_dones = np.array([False for _ in range(self.nenvs)])
        self.buffer_states = model.initial_state
    
        self.plotting_buffer = None
        self.summary_writer = summary_writer
        self.game_num = 0
        self.activate_rr_model = False
        self.transitions_from_lesson_buffer = 0
        
        self.last_lstm_loss = None
        self.last_lstm_last_timestep_loss = None
        self.rel_error = 0

        self.n_pretrain_games = n_pretrain_games
        self.rnd_gen = rnd_gen

        # Replay buffer for RR
        lesson_replay_buffer_type = lessons_buffer_config['type']
        get_n_rr_buffer_games = None
        if lesson_replay_buffer_type == 'decay':
            def get_n_rr_buffer_games(game_num, *args, **kwargs):
                decayed = (lessons_buffer_config['start'] -
                           np.round(game_num / lessons_buffer_config['updates_span'] *
                                    (lessons_buffer_config['start'] - lessons_buffer_config['end'])))
                return int(decayed)
        elif lesson_replay_buffer_type == 'constant':
            def get_n_rr_buffer_games(*args, **kwargs):
                return int(lessons_buffer_config['n_replay_updates'])
        elif lesson_replay_buffer_type == 'off':
            pass
        else:
            raise NotImplementedError("lesson_replay_buffer_type {} not implemented".format(lesson_replay_buffer_type))

        if not (lesson_replay_buffer_type == 'off'):
            lesson_replay_buffer = LessonReplayBuffer(lessons_buffer_config['buffer_size'], rnd_gen=rnd_gen)
        else:
            lesson_replay_buffer = None
        
        self.train_a2c_on_lesson_buffer = lessons_buffer_config['traina2c']
        self.get_n_rr_buffer_games = get_n_rr_buffer_games
        self.lesson_replay_buffer = lesson_replay_buffer
        self.rr_junksize = reward_redistribution_config['rr_junksize']
        self.calc_reward_redistribution = reward_redistribution_config['use_reward_redistribution']
        
        self.avg_reward = 1e-5
        
        self.observation_network_config = observation_network_config
        self.lstm_network_config = lstm_network_config
        self.training_config = training_config
        self.exploration_config = exploration_config
        self.lessons_buffer_config = lessons_buffer_config
        
        self.verbose = verbose
        
        # Broadcasting for exploration
        self.exploration_dict = dict(
            prev_actions=np.zeros((nenv,), dtype=np.int64),
            prev_action_count=np.zeros((nenv,), dtype=np.int64),
            exploration_timesteps=np.array(self.rnd_gen.randint(low=0, high=100, size=(nenv,)), dtype=np.float32),
            exploration_durations=np.array(self.rnd_gen.randint(low=0, high=100, size=(nenv,)), dtype=np.float32),
            avg_game_len=np.array([10], dtype=np.float32),
            gamelengths=np.zeros((nenv,), dtype=np.float32)
        )
        
    def check_action_repeats(self, new_actions):
        """Count how often in a row the same action has been taken"""
        self.exploration_dict['prev_action_count'] *= self.exploration_dict['prev_actions'] == new_actions
        self.exploration_dict['prev_action_count'] += self.exploration_dict['prev_actions'] == new_actions
        self.exploration_dict['prev_actions'][:] = new_actions
        
    def redistribute_reward(self, env_i):
        """Perform reward redistribution on a completed game in self.buffered_incomplete_games, train on lessons
        buffer, and show new episode to lessons buffer;
        
        Parameters
        -------
        env_i : int
            Environment number with completed game
        """
        plotlog = env_i == 0
        calc_reward_redistribution = self.calc_reward_redistribution
        use_reward_redistribution = self.game_num >= self.n_pretrain_games
        
        #
        # Train LSTM on lesson buffer sample, then show lessons buffer episode to A2C
        #
        if self.lesson_replay_buffer is not None and self.activate_rr_model:
            n_rr_buffer_games = self.get_n_rr_buffer_games(game_num=self.game_num)
            if self.lesson_replay_buffer.get_buffer_len() >= n_rr_buffer_games:
                
                # Train reward redistribution model
                rr_buffer_time = time.time()
                print("\treplaying {} RR buffer games...".format(n_rr_buffer_games), end='')
                rr_buffer_steps = 0
                for _ in range(n_rr_buffer_games):
                    rr_sample = self.lesson_replay_buffer.get_sample()
                    rr_temp_dict = self.rr_model.reward_redistribution_junked(
                            tf_session=self.sess, states=rr_sample['states'], actions=rr_sample['actions'],
                            rewards=rr_sample['original_rewards'], avg_reward=self.avg_reward,
                            redistribute_reward=self.train_a2c_on_lesson_buffer and calc_reward_redistribution,
                            use_reward_redistribution=use_reward_redistribution, update=True,
                            details=False, summaries=False, junksize=self.rr_junksize, verbose=False)
                    
                    rr_buffer_steps += len(rr_sample['original_rewards'])
                    self.lesson_replay_buffer.update_sample_loss(loss=rr_temp_dict['rr_loss'], id=rr_sample['id'])
                    
                    # Add lessons buffer episode to queue for policy network (add to queue for agent with min.
                    # transitions in buffer)
                    if self.game_num > self.n_pretrain_games and self.train_a2c_on_lesson_buffer:
                        lesson_buffer_game = [[rr_sample['states'][0, t], rr_sample['actions'][0, t],
                                               rr_sample['neglogpacs'][t], rr_sample['dones'][t],
                                               dict(), rr_sample['original_rewards'][t],
                                               rr_temp_dict['redistributed_reward'][t],
                                               rr_temp_dict['rr_quality'][t]]
                                              for t in range(rr_sample['dones'].shape[0])]
                        completed_game_lens = [len(g) for g in self.buffered_complete_games]
                        completed_game_lens[env_i] += len(self.buffered_incomplete_games[env_i])
                        self.transitions_from_lesson_buffer += len(self.buffered_incomplete_games[env_i])
                        lesson_buffer_game_env_i = np.argmin(completed_game_lens)
                        self.buffered_complete_games[lesson_buffer_game_env_i] += lesson_buffer_game
                if self.verbose:
                    print("lessons buffer losses: {} (training took {} for {} steps)".format(
                            self.lesson_replay_buffer.get_losses(), time.time() - rr_buffer_time, rr_buffer_steps))
        
        #
        # Reward redistribution
        #
        completed_game = list(self.buffered_incomplete_games[env_i])
        self.buffered_incomplete_games[env_i] = []
        actions = list(zip(*completed_game))[1]
        neglogpacs = list(zip(*completed_game))[2]
        dones = list(zip(*completed_game))[3]
        # infos = list(zip(*completed_game))[4]
        original_rewards = list(zip(*completed_game))[5]
        
        rr_states = np.expand_dims(np.stack(list(zip(*completed_game))[0], axis=0), axis=0)
        rr_actions = np.expand_dims(np.stack(actions, axis=0), axis=0)
        rr_rewards = np.stack(original_rewards, axis=0)
        
        rr_dict = dict()
        if self.activate_rr_model:
            rr_dict = self.rr_model.reward_redistribution_junked(tf_session=self.sess,
                                                                 states=rr_states,
                                                                 actions=rr_actions,
                                                                 rewards=rr_rewards,
                                                                 avg_reward=self.avg_reward,
                                                                 redistribute_reward=calc_reward_redistribution,
                                                                 use_reward_redistribution=use_reward_redistribution,
                                                                 update=True, details=plotlog,
                                                                 summaries=plotlog,
                                                                 junksize=self.rr_junksize)
        
        rr_dict['actions'] = actions
        rr_dict['original_rewards'] = original_rewards

        self.avg_reward = 0.99 * self.avg_reward + 0.01 * np.sum(original_rewards)
        self.exploration_dict['avg_game_len'] = (0.99 * self.exploration_dict['avg_game_len']
                                                 + 0.01 * len(original_rewards))

        #
        # Show sample to lesson buffer
        #
        if self.lesson_replay_buffer is not None and self.activate_rr_model:
            self.lesson_replay_buffer.consider_adding_sample(
                    dict(loss=[rr_dict['rr_loss']], states=rr_states, actions=rr_actions,
                         neglogpacs=np.stack(neglogpacs, axis=0), original_rewards=rr_dict['original_rewards'],
                         redistributed_reward=rr_dict['redistributed_reward'], rr_quality=rr_dict['rr_quality'],
                         dones=np.stack(dones, axis=0)))
        
        #
        # Add episode to policy network queue and do some logging
        #
        if self.activate_rr_model and self.verbose:
            print("game_num {} rr_loss: {}".format(self.game_num, rr_dict.get('rr_loss', None)))
        self.last_lstm_loss = rr_dict.get('rr_loss', None)
        self.last_lstm_last_timestep_loss = rr_dict.get('rr_loss_last_timestep', None)
        
        self.rel_error = self.rel_error * 0.99 + 0.01 * rr_dict.get('rel_error', 0)
        
        if (self.game_num > self.n_pretrain_games) and self.activate_rr_model:
            completed_game = [[cg[0], rr_actions[0, t], cg[2], cg[3], cg[4], cg[5],
                               rr_dict['redistributed_reward'][t], rr_dict['rr_quality'][t]]
                              for t, cg in enumerate(completed_game)]
        else:
            completed_game = [[cg[0], rr_actions[0, t], cg[2], cg[3], cg[4], cg[5],
                               original_rewards[t], np.zeros_like(original_rewards[t])]
                              for t, cg in enumerate(completed_game)]
        
        self.buffered_complete_games[env_i] += completed_game
        
        if plotlog and self.activate_rr_model:
            self.plotting_buffer = [rr_dict] + completed_game
            self.summary_writer.add_summary(rr_dict['all_summaries'], self.game_num)
        
        self.game_num += 1
    
    def next_step(self):
        """Get next step from environment; Since reward redistribution is used, games have to be pre-played and buffered
        until all agents have at least 1 complete game and rewards can be redistributed"""
        # Play and buffer games until all agents have a fully buffered game
        while any([len(g) == 0 for g in self.buffered_complete_games]):
            exploration_timesteps = self.exploration_dict['exploration_timesteps']
            avg_game_len = self.exploration_dict['avg_game_len']
            exploration_durations = self.exploration_dict['exploration_durations']
            prev_actions = self.exploration_dict['prev_actions']
            gamelengths = self.exploration_dict['gamelengths']
            
            keep_prev_action = np.asarray(np.logical_and((prev_actions % 5) != 0, prev_actions > 0), dtype=np.float32)
            actions, self.buffer_actor_states, neglogpac = \
                self.model.act_model.action_exploration(self.buffer_actor_obs, self.buffer_actor_states,
                                                        self.buffer_actor_dones,
                                                        exploration_timesteps=exploration_timesteps,
                                                        prev_actions=prev_actions, gamelengths=gamelengths,
                                                        exploration_durations=exploration_durations,
                                                        keep_prev_action=keep_prev_action,
                                                        prev_action_count=self.exploration_dict['prev_action_count'])
            self.check_action_repeats(actions)
            
            old_obs = [self.buffer_actor_obs[env_i].copy() for env_i in range(self.nenvs)]
            self.buffer_actor_obs[:], rews, self.buffer_actor_dones, infos = self.env.step(actions)
            self.exploration_dict['gamelengths'] += 1
            self.exploration_dict['gamelengths'] *= ~self.buffer_actor_dones
            
            for env_i in range(self.nenvs):
                self.buffered_incomplete_games[env_i].append([old_obs[env_i], actions[env_i],
                                                              neglogpac[env_i], self.buffer_actor_dones[env_i],
                                                              infos[env_i], rews[env_i]])
            
            for env_i in np.where(self.buffer_actor_dones)[0]:
                # If an incomplete game finishes, perform reward redistribution and add it to completed games buffer
                self.redistribute_reward(env_i)
                # Set up exploration for next game
                if self.exploration_config['temporal_safe_exploration']:
                    exploration_timesteps[env_i] = np.random.triangular(0, avg_game_len, avg_game_len, 1)
                    exploration_durations[env_i] = self.rnd_gen.randint(low=1, high=avg_game_len)
                else:
                    exploration_timesteps[env_i] = 0
                    exploration_durations[env_i] = 9999999
        
        # Retrieve a timestep from buffered games
        results = [self.buffered_complete_games[env_i].pop(0) for env_i in range(self.nenvs)]
        obs, actions, neglogpacs, dones, infos, original_rewards, mod_rewards, rr_quality = zip(*results)
        obs = np.stack(obs, axis=0)
        return obs, actions, neglogpacs, dones, infos, original_rewards, mod_rewards, rr_quality
    
    def run(self):
        (mb_obs, mb_rewards, mb_actions, mb_values, mb_lstm_values, mb_dones, mb_neglogpacs, mb_original_rewards,
         mb_rr_quality) = [], [], [], [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            (self.obs[:], old_actions, old_neglogpacs, self.dones[:], infos, original_rewardss, rewards,
             rr_quality) = self.next_step()
            new_actions, values, self.states, new_neglogpacs = self.model.step(self.obs, self.states, self.dones)
            actions = old_actions
            neglogpacs = old_neglogpacs
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_rr_quality.append(rr_quality)
            mb_dones.append(self.dones)
            for env_i, info in enumerate(infos):
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append((env_i, maybeepinfo))
            mb_rewards.append(rewards)
            mb_original_rewards.append(original_rewardss)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_original_rewards = np.asarray(mb_original_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_rr_quality = np.asarray(mb_rr_quality, dtype=np.float32)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_original_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_rewards, mb_dones, mb_actions, mb_values, mb_neglogpacs,
                            mb_rr_quality)),
                mb_states, epinfos)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


class StateSaver(object):
    def __init__(self, n_savefiles, state_dir, lesson_replay_buffer=None):
        """Save tensorflow variables, lessons buffer, and general states"""
        self.n_savefiles = n_savefiles
        self.savefiles = list()
        self.state_dir = state_dir
        self.state_path = os.path.join(state_dir, 'state')
        self.file_extension = '.pkl.zip'
        self.lesson_replay_buffer = lesson_replay_buffer
    
    def save(self, state, step):
        """Save lessons buffer and general states and delete old saves if exceeding n_savefiles"""
        # Save numpy buffer
        state_path = self.state_path
        self.savefiles.append(step)
        if state is not None:
            relatively_safe_pickle_dump(state, state_path + '-{}.pkl.zip'.format(step), compression=True)
        if len(self.savefiles) > self.n_savefiles:
            os.remove(state_path + '-{}.pkl.zip'.format(self.savefiles[0]))
            if self.lesson_replay_buffer is not None:
                os.remove(state_path + '-{}.h5py'.format(self.savefiles[0]))
            del self.savefiles[0]
        # Save RR h5py buffer
        if self.lesson_replay_buffer is not None:
            self.lesson_replay_buffer.buffer_to_file(state_path + '-{}.h5py'.format(step))
    
    def separate_save(self, state, filename):
        """Save lessons buffer and general states and do not delete old saves"""
        # Save numpy buffer
        state_path = os.path.join(self.state_dir, filename + '.pkl.zip')
        if state is not None:
            relatively_safe_pickle_dump(state, state_path, compression=True)
        # Save RR h5py buffer
        if self.lesson_replay_buffer is not None:
            state_path = os.path.join(self.state_path, filename + '.h5py')
            self.lesson_replay_buffer.buffer_to_file(state_path)


def load_state(state_path):
    """Load lessons buffer and general states from file"""
    state = pickle_load(state_path, compression=True)
    return state


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          tf_session=None, working_dir='temp', config=None, plotting=None, rnd_gen=None):
    
    plotting_interval = config.get_value('plot_at', 0)
    save_interval = config.get_value('save_at', 0)
    n_savefiles = config.get_value('n_savefiles', 1)
    save_movie = plotting['save_movie']
    save_subplots_line_plots = plotting['save_subplots_line_plots']
    save_subplots = plotting['save_subplots']
    
    summary_writer = tf.summary.FileWriter(working_dir, graph=tf.get_default_graph())
    
    save_dir = os.path.join(working_dir, 'saves')
    make_sure_path_exists(save_dir)
    make_sure_path_exists(os.path.join(save_dir, 'best'))
    
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)
    
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    
    bl_config = config.get_value('bl_config')
    rudder_config = config.get_value('rudder_config')
    reward_redistribution_config = rudder_config['reward_redistribution_config']
    observation_network_config = rudder_config['observation_network_config']
    lstm_network_config = rudder_config['lstm_network_config']
    training_config = rudder_config['training_config']
    exploration_config = rudder_config['exploration_config']
    lessons_buffer_config = rudder_config['lessons_buffer_config']
    
    # ------------------------------------------------------------------------------------------------------------------
    #  Setting up PPO model
    # ------------------------------------------------------------------------------------------------------------------

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm, env=env,
                               reward_redistribution_config=reward_redistribution_config,
                               observation_network_config=observation_network_config,
                               lstm_network_config=lstm_network_config, training_config=training_config,
                               exploration_config=exploration_config)
    model = make_model()
    temperature = [v for v in tf.global_variables() if v.name.find("temperature") != -1][0]
    temperature_pl = tf.placeholder(shape=(), dtype=tf.float32)
    set_temperature_tensor = tf.assign(temperature, temperature_pl)
    
    def set_temperature(temp):
        tf_session.run(set_temperature_tensor, feed_dict={temperature_pl: temp})
    
    # ------------------------------------------------------------------------------------------------------------------
    #  Setting up reward redistribution model
    # ------------------------------------------------------------------------------------------------------------------
    
    rr_model = RewardRedistributionModel(reward_redistribution_config=reward_redistribution_config,
                                         observation_network_config=observation_network_config,
                                         lstm_network_config=lstm_network_config,
                                         training_config=training_config, scopename="RR")
    rr_model.build_model(state_shape=list(env.observation_space.shape), policy_model=model.simple_model,
                         n_actions=env.action_space.n)
    
    # ------------------------------------------------------------------------------------------------------------------
    #  Runner
    # ------------------------------------------------------------------------------------------------------------------
    
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, rr_model=rr_model,
                    summary_writer=summary_writer, n_pretrain_games=training_config['n_pretrain_games'],
                    reward_redistribution_config=reward_redistribution_config,
                    observation_network_config=observation_network_config,
                    lstm_network_config=lstm_network_config, training_config=training_config,
                    exploration_config=exploration_config, lessons_buffer_config=lessons_buffer_config,
                    rnd_gen=rnd_gen)
    
    epinfobuf = deque(maxlen=200)
    tfirststart = time.time()
    
    # ------------------------------------------------------------------------------------------------------------------
    #  Initialize/load variables
    # ------------------------------------------------------------------------------------------------------------------
    print("Initializing variables")
    
    lstm_net_variables = tf.global_variables(scope='RR/lstmnet/')
    tf.variables_initializer(lstm_net_variables).run(session=tf_session)
    
    rr_update_variables = tf.global_variables(scope='rr_update')
    tf.variables_initializer(rr_update_variables).run(session=tf_session)
    
    emean_reward_variable = tf.global_variables(scope='RR/emean_reward')
    tf.variables_initializer(emean_reward_variable).run(session=tf_session)
    
    observations_variable = tf.global_variables(scope='RR/observations')
    tf.variables_initializer(observations_variable).run(session=tf_session)
    
    beta1_variable = tf.global_variables(scope='beta1_power')
    tf.variables_initializer(beta1_variable).run(session=tf_session)
    
    beta2_variable = tf.global_variables(scope='beta2_power')
    tf.variables_initializer(beta2_variable).run(session=tf_session)
    
    rr_preproc_variable = tf.global_variables(scope='RR/rr_visionsystem/')
    tf.variables_initializer(rr_preproc_variable).run(session=tf_session)
    
    model_variables = tf.global_variables(scope='model')
    tf.variables_initializer(model_variables).run(session=tf_session)
    
    value_function_variable = tf.global_variables(scope='RR/lstm_value_function')
    tf.variables_initializer(value_function_variable).run(session=tf_session)
    
    print("uninitialized_variables", tf_session.run(tf.report_uninitialized_variables()))
    
    # Get kernels in first layer for plotting
    policy_kernel_tensors = [p[:, :, -1, :] for p in tf.trainable_variables() if p.name.find('model/c1/w') > -1]
    policy_kernel_tensors_f = [p[:, :, -1, :]
                               for p in tf.trainable_variables() if p.name.find('model/cf1/W_conv') > -1]
    policy_kernel_tensors_d = [p[:, :, -1, :]
                               for p in tf.trainable_variables() if p.name.find('model/cd1/W_conv') > -1]
    rr_kernel_tensors_f = [p[:, :, -1, :] for p in tf.trainable_variables()
                           if p.name.find('RR/rr_visionsystem/cf1/W_conv') > -1]
    rr_kernel_tensors_d = [p[:, :, -1, :] for p in tf.trainable_variables()
                           if p.name.find('RR/rr_visionsystem/cd1/W_conv') > -1]
    
    #
    # Create variable saver and load weights if load_file is specified
    #
    tf_saver = tf.train.Saver(max_to_keep=n_savefiles)
    tf_saver_separate = tf.train.Saver(max_to_keep=10000)
    tf_saver_bestrun = tf.train.Saver(max_to_keep=10)
    state_saver = StateSaver(n_savefiles=n_savefiles, state_dir=save_dir,
                             lesson_replay_buffer=runner.lesson_replay_buffer)
    state_saver_bestrun = StateSaver(n_savefiles=10, state_dir=os.path.join(save_dir, 'best'),
                                     lesson_replay_buffer=runner.lesson_replay_buffer)
    
    trainingstate = {}
    
    if len(config.get_value('load_file_dict', {})):
        print("Starting loading of weights...")
        load_dict = config.get_value('load_file_dict')
        for scope in load_dict.keys():
            if scope == 'states':
                print("\t loading non-tensorflow states from {}".format(load_dict[scope]))
                trainingstate = load_state(load_dict[scope])
            elif scope == 'rr_buffer':
                print("\t loading rr_buffer from {}".format(load_dict[scope]))
                runner.lesson_replay_buffer.buffer_from_file(load_dict[scope])
            else:
                print("\t loading weights: {} -> {}".format(load_dict[scope], scope))
                sys.stdout.flush()
                restore_list = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)]
                restore_saver = tf.train.Saver(var_list=restore_list)
                restore_saver.restore(tf_session, load_dict[scope])
    runner.game_num = trainingstate.get('game_num', 0)
    
    curr_update_n = trainingstate.get('update', 0)
    curr_num_iters = trainingstate.get('num_iters', 1)
    runner.transitions_from_lesson_buffer = trainingstate.get('transitions_from_lesson_buffer', 0)
    
    logger.Logger.CURRENT.output_formats[0].step = curr_update_n
    print("Num step set to ", curr_update_n)
    
    #
    # Finalize graph
    #  This makes our tensorflow graph read-only and prevents further additions to the graph
    #
    tf_session.graph.finalize()
    if tf_session.graph.finalized:
        print("Graph is finalized!")
    else:
        raise ValueError("Could not finalize graph!")
    
    nupdates = total_timesteps // nbatch
    update = curr_update_n
    never_saved = True
    best_return = np.nan

    #
    # Enter training loop
    #
    try:
        while total_timesteps > update * nbatch - runner.transitions_from_lesson_buffer:
            update += 1
            runner.activate_rr_model = update >= training_config['n_no_rr_updates']
            
            assert nbatch % nminibatches == 0
            nbatch_train = nbatch // nminibatches
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = lr(frac)
            cliprangenow = cliprange(frac)
            if bl_config['temperature_decay']:
                set_temperature(temp=np.clip(100 / runner.exploration_dict['avg_game_len'][0], a_min=1e-6, a_max=1))
            
            (obs, returns, mod_rewards, masks, actions, values, neglogpacs, rr_quality, states,
             epinfos) = runner.run()  # pylint: disable=E0632
            
            epinfobuf.extend([epinfo[1] for epinfo in epinfos])
            mblossvals = []
            if states is None: # nonrecurrent version
                inds = np.arange(nbatch)
                for _ in range(noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, mod_rewards, masks, actions, values,
                                                          neglogpacs, rr_quality))
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices))
            else: # recurrent version
                assert nenvs % nminibatches == 0
                envinds = np.arange(nenvs)
                flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
                envsperbatch = nbatch_train // nsteps
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = (arr[mbflatinds] for arr in (obs, returns, mod_rewards, masks, actions, values,
                                                              neglogpacs, rr_quality))
                        mbstates = states[mbenvinds]
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, states=mbstates))
                        
            lossvals = np.mean(mblossvals, axis=0)
            tnow = time.time()
            fps = int(nbatch / (tnow - tstart))
            if update % log_interval == 0 or update == 1:
                print("###############\n", config.specs, "\n", config.working_dir, "\n###############3")
                ev = explained_variance(values, returns)
                logger.logkv("serial_timesteps", update * nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update * nbatch - runner.transitions_from_lesson_buffer)
                logger.logkv("fps", fps)
                logger.logkv("game_num", runner.game_num)
                logger.logkv("explained_variance", float(ev))
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv('eprewmean100', safemean([epinfo['r'] for epinfo in list(epinfobuf)[-100:]]))
                logger.logkv('eplenmean100', safemean([epinfo['l'] for epinfo in list(epinfobuf)[-100:]]))
                logger.logkv('time_elapsed', tnow - tfirststart)
                logger.logkv('Relative_emavg_error_LSTM', runner.rel_error)
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger.logkv(lossname, lossval)
                logger.dumpkvs()

            #
            # Plotting
            #
            if (plotting_interval and ((update % plotting_interval == 0) or update == 1)
                and getattr(runner, 'plotting_buffer', None) is not None and runner.activate_rr_model):
                # Take a complete game from buffer
                game = runner.plotting_buffer
                rr_dict = game[0]
                game = game[1:]
                # until done - store debug in evn0
                game = list(zip(*game))
                frames = [np.array(g[:, :, -1], dtype=np.uint8) for g in game[0]]
                actions = np.asarray(rr_dict['actions'])
                rewards = np.asarray(rr_dict['original_rewards'])
                new_rewards = np.asarray(rr_dict['redistributed_reward'])
                intgrd_from_lstm = np.asarray(rr_dict['intgrd_from_lstm'])
                
                # Prepare some visual indication for redistributed reward
                frames = np.pad(frames, pad_width=((0, 0), (0, 12), (0, 0)), mode='constant')
                for f_i in range(len(intgrd_from_lstm)):
                    r = int(new_rewards[f_i] * 20)
                    height = slice(-12, -9)
                    lenght = slice(15, 15 + abs(r))
                    frames[f_i, height, 10:15] = int(255. * 0.75)
                    if r < 0:
                        frames[f_i, height, lenght] = int(255. * 0.5)
                    elif r > 0:
                        frames[f_i, height, lenght] = 255
        
                    r = int(intgrd_from_lstm[f_i] * 20)
                    height = slice(-8, -5)
                    lenght = slice(15, 15 + abs(r))
                    frames[f_i, height, 10:15] = int(255. * 0.75)
                    if r < 0:
                        frames[f_i, height, lenght] = int(255. * 0.5)
                    elif r > 0:
                        frames[f_i, height, lenght] = 255
        
                    r = int(rewards[f_i] * 20)
                    height = slice(-4, -1)
                    lenght = slice(15, 15 + abs(r))
                    frames[f_i, height, 10:15] = int(255. * 0.75)
                    if r < 0:
                        frames[f_i, height, lenght] = int(255. * 0.5)
                    elif r > 0:
                        frames[f_i, height, lenght] = 255
                
                frames[0, 0, 0] = 255
                frames[0, 0, 1] = 0
                save_movie(images=[np.array(f) for f in frames],
                           filename=os.path.join(working_dir, "states_u{}_g{}.mp4".format(update, runner.game_num)),
                           interval=5, fps=10)
                del frames
                
                save_subplots_line_plots(
                        images=[rewards, new_rewards, rr_dict['intgrd_from_lstm'], actions],
                        subfigtitles=['rewards', 'new_reward', 'intgrd_from_lstm', 'taken actions'],
                        automatic_positioning=True, tight_layout=True,
                        filename=os.path.join(working_dir, "actions_rewards_u{}_g{}.png".format(update,
                                                                                                runner.game_num)))
                
                plot_keys = ['intgrd_from_lstm']
                save_subplots_line_plots(
                        images=([rewards, rr_dict['aux_target_all_ts'][..., 0], rr_dict['predictions'][..., 1],
                                 rr_dict['aux_target_all_ts'][..., 1], rr_dict['predictions'][..., 2],
                                 rr_dict['predictions'][..., 3], rr_dict['predictions'][..., 0],]
                                + [rr_dict.get(k, None) for k in plot_keys if k in plot_keys]),
                        subfigtitles=(['rewards', 'aux1', 'pred_aux1', 'aux2', 'pred_aux2', 'pred_aux3',
                                       'pred_acc_rew'] + plot_keys),
                        automatic_positioning=True, tight_layout=True,
                        filename=os.path.join(working_dir, "reward_details_u{}_g{}.png".format(update,
                                                                                               runner.game_num)))
                
                save_subplots_line_plots(
                        images=[rr_dict['lstm_internals'][:, :, 0], rr_dict['lstm_internals'][:, :, 1],
                                rr_dict['lstm_internals'][:, :, 2], rr_dict['lstm_internals'][:, :, 3],
                                rr_dict['lstm_internals'][:, :, 4], rr_dict['lstm_h'][0, :, :]],
                        subfigtitles=['rr_lstm_ig', 'rr_lstm_og', 'rr_lstm_ci', 'rr_lstm_fg', 'rr_lstm_c',
                                      'rr_lstm_h'],
                        automatic_positioning=True, tight_layout=True,
                        filename=os.path.join(working_dir, "rr_lstm_intern_u{}_g{}.png".format(update,
                                                                                               runner.game_num)))
                
                save_subplots_line_plots(
                        images=[rr_dict['lstm_internals'][:, 0, 0], rr_dict['lstm_internals'][:, 0, 1],
                                rr_dict['lstm_internals'][:, 0, 2], rr_dict['lstm_internals'][:, 0, 3],
                                rr_dict['lstm_internals'][:, 0, 4], rr_dict['lstm_h'][0, :, 0]],
                        subfigtitles=['rr_lstm_ig', 'rr_lstm_og', 'rr_lstm_ci', 'rr_lstm_fg', 'rr_lstm_c', 'rr_lstm_h'],
                        automatic_positioning=True, tight_layout=True,
                        filename=os.path.join(working_dir, "rr_single_lstm_intern_u{}_g{}.png".format(update,
                                                                                                      runner.game_num)))
                
                if len(policy_kernel_tensors):
                    kernels = tf_session.run(policy_kernel_tensors)[0]
                    plot_keys = ['kernel{}'.format(i) for i in range(kernels.shape[-1])]
                    save_subplots(images=[kernels[:, :, i] for i in range(kernels.shape[-1])],
                                  subfigtitles=plot_keys, automatic_positioning=True, tight_layout=True,
                                  filename=os.path.join(working_dir,
                                                        "policy_kernels_u{}_g{}.png".format(update, runner.game_num)))

                if len(policy_kernel_tensors_f):
                    kernels = tf_session.run(policy_kernel_tensors_f)[0]
                    plot_keys = ['kernel{}'.format(i) for i in range(kernels.shape[-1])]
                    save_subplots(images=[kernels[:, :, i] for i in range(kernels.shape[-1])],
                                  subfigtitles=plot_keys, automatic_positioning=True, tight_layout=True,
                                  filename=os.path.join(working_dir,
                                                        "policy_fkernels_u{}_g{}.png".format(update,
                                                                                             runner.game_num)))

                if len(policy_kernel_tensors_d):
                    kernels = tf_session.run(policy_kernel_tensors_d)[0]
                    plot_keys = ['kernel{}'.format(i) for i in range(kernels.shape[-1])]
                    save_subplots(images=[kernels[:, :, i] for i in range(kernels.shape[-1])],
                                  subfigtitles=plot_keys, automatic_positioning=True, tight_layout=True,
                                  filename=os.path.join(working_dir,
                                                        "policy_dkernels_u{}_g{}.png".format(update,
                                                                                             runner.game_num)))

                if len(rr_kernel_tensors_f):
                    kernels = tf_session.run(rr_kernel_tensors_f)[0]
                    plot_keys = ['kernel{}'.format(i) for i in range(kernels.shape[-1])]
                    save_subplots(images=[kernels[:, :, i] for i in range(kernels.shape[-1])],
                                  subfigtitles=plot_keys, automatic_positioning=True, tight_layout=True,
                                  filename=os.path.join(working_dir,
                                                        "rr_fkernels_u{}_g{}.png".format(update,
                                                                                         runner.game_num)))

                if len(rr_kernel_tensors_d):
                    kernels = tf_session.run(rr_kernel_tensors_d)[0]
                    plot_keys = ['kernel{}'.format(i) for i in range(kernels.shape[-1])]
                    save_subplots(images=[kernels[:, :, i] for i in range(kernels.shape[-1])],
                                  subfigtitles=plot_keys, automatic_positioning=True, tight_layout=True,
                                  filename=os.path.join(working_dir,
                                                        "rr_dkernels_u{}_g{}.png".format(update,
                                                                                         runner.game_num)))
            
            #
            # Save the model, lessons buffer, and general state
            #
            trainingstate['game_num'], trainingstate['update'], trainingstate['transitions_from_lesson_buffer'] = \
                runner.game_num, update, runner.transitions_from_lesson_buffer
            trainingstate['num_iters'] = update*nbatch - runner.transitions_from_lesson_buffer
            if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
                state_saver.save(state=trainingstate, step=int(update))
                tf_saver.save(sess=tf_session, save_path=os.path.join(save_dir, "checkpoint"), global_step=int(update))
            if (update % 10000 == 0) and logger.get_dir():
                state_saver.separate_save(state=trainingstate, filename="permanent_save_u{}".format(update))
                tf_saver_separate.save(sess=tf_session,
                                       save_path=os.path.join(save_dir, "permanent_checkpoint_u{}".format(update)))
            if update > 1000:
                if len(epinfobuf):
                    curr_return = max([eb['r'] for eb in epinfobuf])
                    if not np.isfinite(best_return):
                        best_return = curr_return
                        state_saver_bestrun.save(state=trainingstate, step=int(update))
                        tf_saver_bestrun.save(sess=tf_session,
                                              save_path=os.path.join(os.path.join(save_dir, 'best'), "checkpoint"),
                                              global_step=int(update))
                    elif curr_return > best_return:
                        print("New best run! Saving...")
                        best_return = curr_return
                        state_saver_bestrun.save(state=trainingstate, step=int(update))
                        tf_saver_bestrun.save(sess=tf_session,
                                              save_path=os.path.join(os.path.join(save_dir, 'best'), "checkpoint"),
                                              global_step=int(update))
                
        else:
            trainingstate['game_num'], trainingstate['update'] = runner.game_num, update
            state_saver.save(state=trainingstate, step=int(update))
            tf_saver.save(sess=tf_session, save_path=os.path.join(save_dir, "checkpoint"), global_step=int(update))
            env.close()
            # Close and terminate plotting queue and subprocesses
            from TeLL.utility.plotting import terminate_plotting_daemon
            terminate_plotting_daemon()
            tf_session.close()
            print("Done!")
            
    except:
        state_saver.save(state=trainingstate, step=int(update))
        tf_saver.save(sess=tf_session, save_path=os.path.join(save_dir, "checkpoint"), global_step=int(update))
        env.close()
        # Close and terminate plotting queue and subprocesses
        from TeLL.utility.plotting import terminate_plotting_daemon
        terminate_plotting_daemon()
        tf_session.close()
        raise


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def safemean_no_nan(xs, default):
    return default if len(xs) == 0 else np.mean(xs)

