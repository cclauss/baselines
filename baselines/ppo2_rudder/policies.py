# -*- coding: utf-8 -*-
"""policies.py: Adaption of baselines.ppo2.policies.py for RUDDER for atari games

Author -- Michael Widrich
Contact -- widrich@bioinf.jku.at

"""
import numpy as np
import tensorflow as tf

from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.distributions import make_pdtype

from baselines.ppo2_rudder.reward_redistribution import RewardRedistributionModel, observation_network
from TeLL.layers import StopGradientLayer


def nature_cnn(unscaled_images):
    """Convolutional parts of CNN from Nature paper
    
    Taken from baselines.ppo2.policies.py
    
    Parameters
    -------
    unscaled_images : tensorflow tensor
        Frame of shape (batchsize, x, y, c)
    
    Returns
    -------
    tensorflow tensor
        Output features of last convolutional layer with flattened x/y/c dimensions
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return h3


def lstm(xs, ms, s, scope, nh):
    """LSTM layer for policy network, using same weight and bias initialization as LSTM in reward redistribution model
    
    Based on baselines.ppo2.policies.py; These initializations were taken directly from the redistribution model LSTM
    and could be optimized;
    """
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    
    lstm_w_init = lambda scale: lambda *args, **kwargs: tf.truncated_normal(*args, **kwargs) * scale
    truncated_normal_init = lambda mean, stddev: \
        lambda *args, **kwargs: tf.truncated_normal(mean=mean, stddev=stddev, *args, **kwargs)
    
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        wx_ig = tf.get_variable("wx_ig", initializer=lstm_w_init(0.1)([nin, nh]))
        wx_og = tf.get_variable("wx_og", initializer=lstm_w_init(0.1)([nin, nh]))
        wx_ci = tf.get_variable("wx_ci", initializer=lstm_w_init(0.0001)([nin, nh]))
        wx_fg = tf.get_variable("wx_fg", initializer=lstm_w_init(0.1)([nin, nh]))
        
        wh_ig = tf.get_variable("wh_ig", initializer=lstm_w_init(0.001)([nh, nh]))
        wh_og = tf.get_variable("wh_og", initializer=lstm_w_init(0.001)([nh, nh]))
        wh_ci = tf.get_variable("wh_ci", initializer=lstm_w_init(0.001)([nh, nh]))
        wh_fg = tf.get_variable("wh_fg", initializer=lstm_w_init(0.001)([nh, nh]))
        
        b_ig = tf.get_variable("b_ig", initializer=truncated_normal_init(mean=-5, stddev=0.1)([nh]))
        b_fg = tf.get_variable("b_fg", initializer=truncated_normal_init(mean=12, stddev=0.1)([nh]))
        b_og = tf.get_variable("b_og", initializer=truncated_normal_init(mean=-5, stddev=0.1)([nh]))
        b_ci = tf.get_variable("b_ci", initializer=truncated_normal_init(mean=0, stddev=0.1)([nh]))
        
        wx = tf.concat([wx_ig, wx_fg, wx_og, wx_ci], axis=1)
        wh = tf.concat([wh_ig, wh_fg, wh_og, wh_ci], axis=1)
        b = tf.concat([b_ig, b_fg, b_og, b_ci], axis=0)
    
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c *= (1 - m)
        h *= (1 - m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.identity(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s


class LstmPolicy(object):
    def __init__(self, tf_session, ob_space, ac_space, nbatch,
                 reward_redistribution_config, observation_network_config, lstm_network_config, training_config,
                 exploration_config, nsteps, nlstm=64, reuse=False):
        """LSTM policy network, as described in RUDDER paper
        
        Based on baselines.ppo2.policies.py; LSTM layer sees features from it's own trainable observation network and
        the features from the reward redistribution observation network;
        
        Parameters
        -------
        tf_session : tensorflow session
            tensorflow session to compute the graph in
        ob_space
            Baselines ob_space object (see ppo2_rudder.py); must provide .shape attribute for (x, y, c) shapes;
        ac_space
            Baselines ac_space object (see ppo2_rudder.py); must provide .n attribute for number of possible actions;
        nbatch : int
            Batchsize
        nsteps : int
            Fixed number of timesteps to process at once
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
        nlstm : int
            Number of LSTM units (=memory cells)
        reuse : bool
            Reuse tensorflow variables?
        """
        #
        # Shapes
        #
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        seq_ob_shape = (nenv, -1, nh, nw, 1)
        nact = ac_space.n
        
        #
        # Placeholders for inputs
        #
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        
        #
        # Prepare input
        #
        single_frames = tf.cast(tf.reshape(X[..., -1:], shape=seq_ob_shape), dtype=tf.float32)
        delta_frames = single_frames - tf.cast(tf.reshape(X[..., -2:-1], shape=seq_ob_shape), dtype=tf.float32)
        
        #
        #  Get observation features from RR model
        #
        rr_model = RewardRedistributionModel(reward_redistribution_config=reward_redistribution_config,
                                             observation_network_config=observation_network_config,
                                             lstm_network_config=lstm_network_config, training_config=training_config,
                                             scopename="RR")
        self.rr_observation_model = rr_model
        rr_observation_layer = rr_model.get_visual_features(single_frame=single_frames, delta_frame=delta_frames,
                                                            additional_inputs=[])
        
        #
        #  Build policy network
        #
        with tf.variable_scope("model", reuse=reuse):
            temperature = tf.get_variable(initializer=tf.constant(1, dtype=tf.float32), trainable=False,
                                          name='temperature')
            
            additional_inputs = [StopGradientLayer(rr_observation_layer)]
            observation_layers, observation_features = observation_network(
                    single_frame=single_frames, delta_frame=delta_frames, additional_inputs=additional_inputs,
                    observation_network_config=observation_network_config)
            
            self.observation_features_shape = observation_features.get_output_shape()
            
            xs = [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps,
                                                       value=tf.reshape(observation_layers[-1].get_output(),
                                                                        [nenv, nsteps, -1]))]
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            h6 = h5
            pi = fc(h6, 'pi', nact)
            vf = fc(h6, 'v', 1)
        
        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)
        
        if exploration_config['sample_actions_from_softmax']:
            a0 = self.pd.sample_temp(temperature=temperature)
        else:
            a0 = tf.argmax(pi, axis=-1)
        
        v0 = vf[:, 0]
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
        
        def step(ob, state, mask):
            a, v, s, neglogp = tf_session.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})
            return a, v, s, neglogp
        
        def value(ob, state, mask):
            return tf_session.run(v0, {X:ob, S:state, M:mask})
        
        def action(ob, state, mask, *_args, **_kwargs):
            a, s, neglogp = tf_session.run([a0, snew, neglogp0], {X:ob, S:state, M:mask})
            return a, s, neglogp
        
        #
        # Placeholders for exploration
        #
        n_envs = pi.shape.as_list()[0]
        exploration_timesteps_pl = tf.placeholder(dtype=tf.float32, shape=(n_envs,))
        prev_actions_pl = tf.placeholder(dtype=tf.int64, shape=(n_envs,))
        gamelengths_pl = tf.placeholder(dtype=tf.float32, shape=(n_envs,))
        keep_prev_action_pl = tf.placeholder(dtype=tf.bool, shape=(n_envs,))
        prev_action_count_pl = tf.placeholder(dtype=tf.int64, shape=(n_envs,))
        exploration_durations_pl = tf.placeholder(dtype=tf.float32, shape=(n_envs,))
        
        #
        # Setting up safe exploration
        #
        explore = tf.logical_and(tf.logical_and(tf.less_equal(exploration_timesteps_pl, gamelengths_pl),
                                                tf.less_equal(gamelengths_pl,
                                                              exploration_timesteps_pl + exploration_durations_pl)),
                                 tf.not_equal(exploration_timesteps_pl, tf.constant(-1, dtype=tf.float32)))

        safe_pi = pi - tf.reduce_min(pi, axis=-1, keep_dims=True)
        safe_pi /= tf.reduce_max(safe_pi, axis=-1, keep_dims=True)
        save_pi_thresholds = (1 - (tf.expand_dims(tf.range(n_envs, dtype=tf.float32), axis=1)
                                   / (n_envs + (n_envs == 1) - 1)) * (1 - exploration_config['save_pi_threshold']))
        safe_pi = tf.cast(tf.greater_equal(safe_pi, save_pi_thresholds), dtype=tf.float32)
        safe_pi /= tf.reduce_sum(safe_pi)
        
        rand_safe_a = tf.multinomial(safe_pi, 1)[:, 0]
        
        safe_pi_flat = tf.reshape(safe_pi, (-1,))
        prev_action_is_safe = tf.gather(safe_pi_flat,
                                        prev_actions_pl + tf.range(safe_pi.shape.as_list()[0], dtype=tf.int64)
                                        * safe_pi.shape.as_list()[1])
        prev_action_is_safe = tf.greater(prev_action_is_safe, tf.constant(0, dtype=tf.float32))
        
        a_explore = tf.where(tf.logical_and(tf.logical_and(keep_prev_action_pl,
                                                           tf.not_equal(gamelengths_pl, exploration_timesteps_pl)),
                                            prev_action_is_safe),
                             prev_actions_pl, rand_safe_a)
        
        a_explore = tf.where(explore, a_explore, a0)
        
        # Make sure the actor doesn't repeat an action too often (otherwise screensaver might start)
        rand_a = tf.random_uniform(shape=a0.get_shape(), minval=0, maxval=ac_space.n, dtype=a0.dtype)
        a_explore = tf.where(tf.greater(prev_action_count_pl, tf.constant(20, dtype=tf.int64)), rand_a, a_explore)
        
        if not exploration_config['temporal_safe_exploration']:
            a_explore = a0
            
        neglogp_explore = self.pd.neglogp(a_explore)
        
        def action_exploration(ob, state, mask, *_args, exploration_timesteps, prev_actions, gamelengths,
                               keep_prev_action, prev_action_count, exploration_durations, **_kwargs):
            """Get actions with exploration for long-term reward"""
            a, s, neglogp = tf_session.run([a_explore, snew, neglogp_explore],
                                  {X: ob, S:state, M:mask, exploration_timesteps_pl: exploration_timesteps,
                                   prev_actions_pl: prev_actions,
                                   gamelengths_pl: gamelengths, exploration_durations_pl: exploration_durations,
                                   keep_prev_action_pl: keep_prev_action, prev_action_count_pl: prev_action_count})
            return a, s, neglogp
        
        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.action = action
        self.action_exploration = action_exploration
        self.seq_ob_shape = seq_ob_shape
        self.exploration_config = exploration_config
        
    def get_observation_features(self, frame, delta):
        """Get output features of observation network (to be fed into reward redistribution network)"""
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            _, observation_features = observation_network(
                    single_frame=frame[..., -1:], delta_frame=delta, additional_inputs=[],
                    observation_network_config=self.exploration_config['observation_network_config'])
            observation_features = observation_features.get_output()
        
        return observation_features


class LstmPolicyDense(object):
    def __init__(self, tf_session, ob_space, ac_space, nbatch,
                 reward_redistribution_config, observation_network_config, lstm_network_config, training_config,
                 exploration_config, nsteps, nlstm=64, reuse=False):
        """LSTM policy network with additional dense layer after LSTM layer, as described in RUDDER paper
        
        Based on baselines.ppo2.policies.py; LSTM layer sees features from it's own trainable observation network and
        the features from the reward redistribution observation network; The additional dense layer after the LSTM
        layer contains 128 hidden units;
        
        Parameters
        -------
        tf_session : tensorflow session
            tensorflow session to compute the graph in
        ob_space
            Baselines ob_space object (see ppo2_rudder.py); must provide .shape attribute for (x, y, c) shapes;
        ac_space
            Baselines ac_space object (see ppo2_rudder.py); must provide .n attribute for number of possible actions;
        nbatch : int
            Batchsize
        nsteps : int
            Fixed number of timesteps to process at once
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
        nlstm : int
            Number of LSTM units (=memory cells)
        reuse : bool
            Reuse tensorflow variables?
        """
        #
        # Shapes
        #
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        seq_ob_shape = (nenv, -1, nh, nw, 1)
        nact = ac_space.n
        
        #
        # Placeholders
        #
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        
        #
        # Prepare input
        #
        single_frames = tf.cast(tf.reshape(X[..., -1:], shape=seq_ob_shape), dtype=tf.float32)
        delta_frames = single_frames - tf.cast(tf.reshape(X[..., -2:-1], shape=seq_ob_shape), dtype=tf.float32)
        
        #
        #  Get observation features from RR model
        #
        rr_model = RewardRedistributionModel(reward_redistribution_config=reward_redistribution_config,
                                             observation_network_config=observation_network_config,
                                             lstm_network_config=lstm_network_config, training_config=training_config,
                                             scopename="RR")
        self.rr_observation_model = rr_model
        rr_observation_layer = rr_model.get_visual_features(single_frame=single_frames, delta_frame=delta_frames,
                                                            additional_inputs=[])
        
        #
        #  Build policy network
        #
        with tf.variable_scope("model", reuse=reuse):
            temperature = tf.get_variable(initializer=tf.constant(1, dtype=tf.float32), trainable=False,
                                          name='temperature')
            
            additional_inputs = [StopGradientLayer(rr_observation_layer)]
            observation_layers, observation_features = observation_network(
                    single_frame=single_frames, delta_frame=delta_frames, additional_inputs=additional_inputs,
                    observation_network_config=observation_network_config)
            
            self.observation_features_shape = observation_features.get_output_shape()
            
            xs = [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps,
                                                       value=tf.reshape(observation_layers[-1].get_output(),
                                                                        [nenv, nsteps, -1]))]
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            h6 = fc(h5, 'fc1', nh=128, init_scale=np.sqrt(2))
            pi = fc(h6, 'pi', nact)
            vf = fc(h6, 'v', 1)
        
        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)
        
        if exploration_config['sample_actions_from_softmax']:
            a0 = self.pd.sample_temp(temperature=temperature)
        else:
            a0 = tf.argmax(pi, axis=-1)
        
        v0 = vf[:, 0]
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
        
        def step(ob, state, mask):
            a, v, s, neglogp = tf_session.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})
            return a, v, s, neglogp
        
        def value(ob, state, mask):
            return tf_session.run(v0, {X:ob, S:state, M:mask})
        
        def action(ob, state, mask, *_args, **_kwargs):
            a, s, neglogp = tf_session.run([a0, snew, neglogp0], {X:ob, S:state, M:mask})
            return a, s, neglogp
        
        #
        # Placeholders for exploration
        #
        n_envs = pi.shape.as_list()[0]
        exploration_timesteps_pl = tf.placeholder(dtype=tf.float32, shape=(n_envs,))
        prev_actions_pl = tf.placeholder(dtype=tf.int64, shape=(n_envs,))
        gamelengths_pl = tf.placeholder(dtype=tf.float32, shape=(n_envs,))
        keep_prev_action_pl = tf.placeholder(dtype=tf.bool, shape=(n_envs,))
        prev_action_count_pl = tf.placeholder(dtype=tf.int64, shape=(n_envs,))
        exploration_durations_pl = tf.placeholder(dtype=tf.float32, shape=(n_envs,))
        
        #
        # Setting up safe exploration
        #
        explore = tf.logical_and(tf.logical_and(tf.less_equal(exploration_timesteps_pl, gamelengths_pl),
                                                tf.less_equal(gamelengths_pl,
                                                              exploration_timesteps_pl + exploration_durations_pl)),
                                 tf.not_equal(exploration_timesteps_pl, tf.constant(-1, dtype=tf.float32)))

        safe_pi = pi - tf.reduce_min(pi, axis=-1, keep_dims=True)
        safe_pi /= tf.reduce_max(safe_pi, axis=-1, keep_dims=True)
        save_pi_thresholds = (1 - (tf.expand_dims(tf.range(n_envs, dtype=tf.float32), axis=1)
                                   / (n_envs + (n_envs == 1) - 1)) * (1 - exploration_config['save_pi_threshold']))
        safe_pi = tf.cast(tf.greater_equal(safe_pi, save_pi_thresholds), dtype=tf.float32)
        safe_pi /= tf.reduce_sum(safe_pi)
        
        rand_safe_a = tf.multinomial(safe_pi, 1)[:, 0]
        
        safe_pi_flat = tf.reshape(safe_pi, (-1,))
        prev_action_is_safe = tf.gather(safe_pi_flat,
                                        prev_actions_pl + tf.range(safe_pi.shape.as_list()[0], dtype=tf.int64)
                                        * safe_pi.shape.as_list()[1])
        prev_action_is_safe = tf.greater(prev_action_is_safe, tf.constant(0, dtype=tf.float32))
        
        a_explore = tf.where(tf.logical_and(tf.logical_and(keep_prev_action_pl,
                                                           tf.not_equal(gamelengths_pl, exploration_timesteps_pl)),
                                            prev_action_is_safe),
                             prev_actions_pl, rand_safe_a)
        
        a_explore = tf.where(explore, a_explore, a0)
        
        # Make sure the actor doesn't repeat an action too often (otherwise screensaver might start)
        rand_a = tf.random_uniform(shape=a0.get_shape(), minval=0, maxval=ac_space.n, dtype=a0.dtype)
        a_explore = tf.where(tf.greater(prev_action_count_pl, tf.constant(20, dtype=tf.int64)), rand_a, a_explore)
        
        if not exploration_config['temporal_safe_exploration']:
            a_explore = a0
            
        neglogp_explore = self.pd.neglogp(a_explore)
        
        def action_exploration(ob, state, mask, *_args, exploration_timesteps, prev_actions, gamelengths,
                               keep_prev_action, prev_action_count, exploration_durations, **_kwargs):
            """Get actions with exploration for long-term reward"""
            a, s, neglogp = tf_session.run([a_explore, snew, neglogp_explore],
                                  {X: ob, S:state, M:mask, exploration_timesteps_pl: exploration_timesteps,
                                   prev_actions_pl: prev_actions,
                                   gamelengths_pl: gamelengths, exploration_durations_pl: exploration_durations,
                                   keep_prev_action_pl: keep_prev_action, prev_action_count_pl: prev_action_count})
            return a, s, neglogp
        
        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.action = action
        self.action_exploration = action_exploration
        self.seq_ob_shape = seq_ob_shape
        self.exploration_config = exploration_config
        
    def get_observation_features(self, frame, delta):
        """Get output features of observation network (to be fed into reward redistribution network)"""
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            _, observation_features = observation_network(
                    single_frame=frame[..., -1:], delta_frame=delta, additional_inputs=[],
                    observation_network_config=self.exploration_config['observation_network_config'])
            observation_features = observation_features.get_output()
        
        return observation_features


class CnnPolicy(object):
    def __init__(self, tf_session, ob_space, ac_space, nbatch,
                 reward_redistribution_config, observation_network_config, lstm_network_config, training_config,
                 exploration_config, reuse=False, **kwargs):
    
        """CNN policy network, as described in RUDDER paper

        Based on baselines.ppo2.policies.py; Dense layer sees features from it's own trainable observation network and
        the features from the reward redistribution observation network;

        Parameters
        -------
        tf_session : tensorflow session
            tensorflow session to compute the graph in
        ob_space
            Baselines ob_space object (see ppo2_rudder.py); must provide .shape attribute for (x, y, c) shapes;
        ac_space
            Baselines ac_space object (see ppo2_rudder.py); must provide .n attribute for number of possible actions;
        nbatch : int
            Batchsize
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
        reuse : bool
            Reuse tensorflow variables?
        """
        #
        # Shapes
        #
        nh, nw, nc = ob_space.shape
        activ = tf.nn.relu
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        
        #
        # Placeholders
        #
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        
        #
        # Prepare input
        #
        single_frames = tf.cast(tf.expand_dims(X[..., -1:], axis=1), dtype=tf.float32)
        delta_frames = single_frames - tf.cast(tf.expand_dims(X[..., -2:-1], axis=1), dtype=tf.float32)
        delta_frames *= tf.reshape(M, shape=(nbatch, 1, 1, 1, 1))

        #
        #  Get observation features from RR model
        #
        rr_model = RewardRedistributionModel(reward_redistribution_config=reward_redistribution_config,
                                             observation_network_config=observation_network_config,
                                             lstm_network_config=lstm_network_config, training_config=training_config,
                                             scopename="RR")
        self.rr_observation_model = rr_model
        rr_observation_layer = rr_model.get_visual_features(single_frame=single_frames, delta_frame=delta_frames,
                                                            additional_inputs=[])
        # Get output tensor
        rr_observations = rr_observation_layer.get_output()[:, 0]
        
        #
        #  Build policy network
        #
        with tf.variable_scope("model", reuse=reuse):
            temperature = tf.get_variable(initializer=tf.constant(1, dtype=tf.float32), trainable=False,
                                          name='temperature')
            observation_features = nature_cnn(X)
        
        self.observation_features_shape = tf.expand_dims(observation_features, axis=0).shape
        
        #  Concat observation feature from RR model and A2C model
        h_for_a2c = tf.concat([observation_features, tf.stop_gradient(rr_observations)], axis=-1)
        
        with tf.variable_scope("model", reuse=reuse):
            h_for_a2c = activ(fc(h_for_a2c, 'fc1', nh=512, init_scale=np.sqrt(2)))
        with tf.variable_scope("model", reuse=reuse):
            pi = fc(h_for_a2c, 'pi', nact, init_scale=0.01)
            vf = fc(h_for_a2c, 'v', 1)[:, 0]
        
        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        if exploration_config['sample_actions_from_softmax']:
            a0 = self.pd.sample_temp(temperature=temperature)
        else:
            a0 = tf.argmax(pi, axis=-1)
        
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None
        
        def step(ob, state, mask, *_args, **_kwargs):
            a, v, neglogp = tf_session.run([a0, vf, neglogp0], {X: ob, M:mask})
            return a, v, self.initial_state, neglogp
        
        def value(ob, state, mask, *_args, **_kwargs):
            return tf_session.run(vf, {X: ob, M:mask})
        
        def action(ob, state, mask, *_args, **_kwargs):
            a, neglogp = tf_session.run([a0, neglogp0], {X:ob, M:mask})
            return a, self.initial_state, neglogp
        
        #
        # Placeholders for exploration
        #
        n_envs = pi.shape.as_list()[0]
        exploration_timesteps_pl = tf.placeholder(dtype=tf.float32, shape=(n_envs,))
        prev_actions_pl = tf.placeholder(dtype=tf.int64, shape=(n_envs,))
        gamelengths_pl = tf.placeholder(dtype=tf.float32, shape=(n_envs,))
        keep_prev_action_pl = tf.placeholder(dtype=tf.bool, shape=(n_envs,))
        prev_action_count_pl = tf.placeholder(dtype=tf.int64, shape=(n_envs,))
        exploration_durations_pl = tf.placeholder(dtype=tf.float32, shape=(n_envs,))
        
        #
        # Setting up safe exploration
        #
        explore = tf.logical_and(tf.logical_and(tf.less_equal(exploration_timesteps_pl, gamelengths_pl),
                                                tf.less_equal(gamelengths_pl,
                                                              exploration_timesteps_pl + exploration_durations_pl)),
                                 tf.not_equal(exploration_timesteps_pl, tf.constant(-1, dtype=tf.float32)))
        
        safe_pi = pi - tf.reduce_min(pi, axis=-1, keep_dims=True)
        safe_pi /= tf.reduce_max(safe_pi, axis=-1, keep_dims=True)
        save_pi_thresholds = (1 - (tf.expand_dims(tf.range(n_envs, dtype=tf.float32), axis=1)
                                   / (n_envs + (n_envs == 1) - 1)) * (1 - exploration_config['save_pi_threshold']))
        safe_pi = tf.cast(tf.greater_equal(safe_pi, save_pi_thresholds), dtype=tf.float32)
        safe_pi /= tf.reduce_sum(safe_pi)
        
        rand_safe_a = tf.multinomial(safe_pi, 1)[:, 0]

        safe_pi_flat = tf.reshape(safe_pi, (-1,))
        prev_action_is_safe = tf.gather(safe_pi_flat,
                                        prev_actions_pl + tf.range(safe_pi.shape.as_list()[0], dtype=tf.int64)
                                        * safe_pi.shape.as_list()[1])
        prev_action_is_safe = tf.greater(prev_action_is_safe, tf.constant(0, dtype=tf.float32))
        
        a_explore = tf.where(tf.logical_and(tf.logical_and(keep_prev_action_pl,
                                                           tf.not_equal(gamelengths_pl, exploration_timesteps_pl)),
                                            prev_action_is_safe),
                             prev_actions_pl, rand_safe_a)
        
        a_explore = tf.where(explore, a_explore, a0)
        
        # Make sure the actor doesn't repeat an action too often (otherwise screensaver might start)
        rand_a = tf.random_uniform(shape=a0.get_shape(), minval=0, maxval=ac_space.n, dtype=a0.dtype)
        a_explore = tf.where(tf.greater(prev_action_count_pl, tf.constant(20, dtype=tf.int64)), rand_a, a_explore)

        if not exploration_config['temporal_safe_exploration']:
            a_explore = a0

        neglogp_explore = self.pd.neglogp(a_explore)
        
        def action_exploration(ob, state, mask, *_args, exploration_timesteps, prev_actions, gamelengths,
                               keep_prev_action, prev_action_count, exploration_durations, **_kwargs):
            """Exploration for long-term reward"""
            a, neglogp = tf_session.run([a_explore, neglogp_explore],
                                        {X:ob, M:mask, exploration_timesteps_pl:exploration_timesteps,
                                         prev_actions_pl:prev_actions,
                                         gamelengths_pl:gamelengths, exploration_durations_pl:exploration_durations,
                                         keep_prev_action_pl:keep_prev_action, prev_action_count_pl:prev_action_count})
            return a, self.initial_state, neglogp
        
        self.X = X
        self.M = M
        self.pi = pi
        self.vf = vf
        self.step = step
        self.action = action
        self.value = value
        self.action_exploration = action_exploration
        
    def get_observation_features(self, frame, delta=None):
        """Get output features of observation network (to be fed into reward redistribution network)"""
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            return tf.expand_dims(nature_cnn(frame[:, 0]), dim=1)
