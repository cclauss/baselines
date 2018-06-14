# -*- coding: utf-8 -*-
"""reward_redistribution.py: RUDDER return decomposition and reward redistribution model for atari games

Author -- Michael Widrich
Contact -- widrich@bioinf.jku.at

"""
import sys
import time
from collections import OrderedDict
import numpy as np

import tensorflow as tf
from TeLL.layers import (DenseLayer, LSTMLayer, RNNInputLayer, ConcatLayer, MultiplyFactorLayer,
                         ReshapeLayer, SumLayer, LSTMLayerGetNetInput, LSTMLayerSetNetInput, StopGradientLayer)
from TeLL.initializations import constant
from TeLL.utility.misc_tensorflow import layers_from_specs, tensor_shape_with_flexible_dim, TriangularValueEncoding
from TeLL.regularization import regularize


def observation_network(single_frame, delta_frame, additional_inputs, observation_network_config):
    """Frame processing for LSTM-based network
    
    Frame processing for LSTM-based network; single_frame and delta_frame are processed by convolutional layers,
    flattened, and concatenated with the features in additional_inputs;
    
    Parameters
    -------
    single_frame : TeLL layer or tensor
        Single input frame of shape (batchsize, timesteps or 1, x, y, c)
    
    delta_frame : TeLL layer or tensor
        Pixel-wise delta of input frame of shape (batchsize, timesteps or 1, x, y, c)
    
    additional_inputs : list of TeLL layers or tensors
        List of additional inputs of shape (batchsize, timesteps or 1, f)
    
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
    
    Returns
    -------
    observation_layers : list of TeLL layers
        Layers in observation network
    
    visual_features : TeLL layer
        Features created from visual input without additional_inputs
    """
    
    print("Building observation network...")
    observation_layers = list()
    
    #
    # Preprocessing of single input frames
    #
    if observation_network_config['show_states']:
        print("\tSingle frame preprocessing...")
        lstm_states_prepoc_layers = []
    
        # Normalize states to [-1, 1]
        state_scaling_layer = MultiplyFactorLayer(single_frame, factor=tf.constant(2 / 255., dtype=tf.float32))
        lstm_states_prepoc_layers.append(SumLayer([state_scaling_layer, tf.constant([-1.],
                                                                                    dtype=tf.float32)]))
        
        lstm_states_prepoc_layers += layers_from_specs(incoming=lstm_states_prepoc_layers[-1],
                                                       layerspecs=observation_network_config['prepoc_states'])
        observation_layers.append(lstm_states_prepoc_layers[-1])
        
    #
    # Preprocessing of delta input frames
    #
    if observation_network_config['show_statedeltas']:
        print("\tDelta frame preprocessing...")
        lstm_deltas_prepoc_layers = []
    
        # Normalize state deltas to [-1, 1]
        lstm_deltas_prepoc_layers.append(MultiplyFactorLayer(delta_frame, factor=tf.constant(1 / 255.,
                                                                                             dtype=tf.float32)))
    
        lstm_deltas_prepoc_layers += layers_from_specs(incoming=lstm_deltas_prepoc_layers[-1],
                                                       layerspecs=observation_network_config['prepoc_deltas'])
        observation_layers.append(lstm_deltas_prepoc_layers[-1])

    #
    # Further preprocessing of visual observations (concatenated frame- and delta frame features)
    #
    if len(observation_layers) > 1:
        observation_layers.append(ConcatLayer(observation_layers, name="ObservationsConcatLayer"))

    if observation_network_config['show_states'] or observation_network_config['show_statedeltas']:
        print("\tObservations preprocessing...")
        observation_layers += layers_from_specs(incoming=observation_layers[-1],
                                                layerspecs=observation_network_config['prepoc_observations'])
        
        print("\t\tbuilding {}...".format('ReshapeLayer'), end='')
        observation_layers.append(ReshapeLayer(observation_layers[-1],
                                               shape=(observation_layers[-1].get_output_shape()[:2]
                                                      + [np.prod(observation_layers[-1].get_output_shape()[2:])]),
                                               name='ObservationsFlattenLayer'))
        print(" in {} / out {}".format(observation_layers[-2].get_output_shape(),
                                       observation_layers[-1].get_output_shape()))
    visual_features = observation_layers[-1]
    
    #
    # Concatenate observations with additional input features
    #
    if len(additional_inputs):
        print("\tAppending additional inputs...")
        observation_layers.append(ConcatLayer([observation_layers[-1]] + additional_inputs,
                                              name="ConcatObservationsAdditionalFeatures"))
    
    print("Observation network output shape: {}".format(observation_layers[-1].get_output_shape()))
    
    if len(observation_layers) == 0:
        raise ValueError("Observation network empty! Please set show_environment or show_sctaions to True or "
                         "specify additional_inputs!")
        
    return observation_layers, visual_features

    
class RewardRedistributionModel(object):
    def __init__(self, reward_redistribution_config, observation_network_config, lstm_network_config, training_config,
                 scopename="RR", loop_parallel_iterations=200, aux_target_horizont=10,
                 write_histograms=False):
        """LSTM based network for decomposing return and performing reward redistribution via `integrated gradients`_
        as described in RUDDER paper
        
        batchsize != 1 currently not supported; Variable sizes for n_timesteps may be used; Example configurations can
        be found in folder ppo2_rudder/configs;
        
        Parameters
        ----------
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
        scopename : str
            Name for tensorflow variable scope
        loop_parallel_iterations : int
            Number of max. parallel loop computations in tf.while loop
        aux_target_horizont : int
            Number of timesteps to predict ahead in one of the auxiliary tasks (task is to predict the accumulated
            reward in the next aux_target_horizont timesteps).
        write_histograms : bool
            Write histograms of weights and activations to tensorboad for debugging?
        """
        
        self.scopename = scopename
        
        self.placeholders = None
        self.data_tensors = None
        self.operation_tensors = None
        self.summaries = None
        self.loop_parallel_iterations = loop_parallel_iterations
        
        self.aux_target_pad = np.zeros((aux_target_horizont - 1,), dtype=np.float32)
        self.aux_target_filter = np.ones((aux_target_horizont,), dtype=np.float32) / aux_target_horizont
        
        self.reward_redistribution_config = reward_redistribution_config
        self.observation_network_config = observation_network_config
        self.lstm_network_config = lstm_network_config
        self.training_config = training_config
        self.reward_redistribution_config = reward_redistribution_config

        self.ingametime = TriangularValueEncoding(**self.lstm_network_config['timestep_encoding'])
        
        self.write_histograms = write_histograms
    
    def get_visual_features(self, single_frame, delta_frame, additional_inputs):
        """Get output features of observation network only"""
        print("Building RR visual observation network...")
        with tf.variable_scope(self.scopename, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("rr_visionsystem", reuse=tf.AUTO_REUSE):
                # Create first part of observation network and get visual_features to feed to A2C
                _, visual_features = observation_network(single_frame=single_frame,
                                                         delta_frame=delta_frame,
                                                         additional_inputs=additional_inputs,
                                                         observation_network_config=self.observation_network_config)
            
        return visual_features
    
    def build_model(self, state_shape, policy_model, n_actions):
        """Build reward redistribution network, including observation network and LSTM network"""
        
        ingametime = self.ingametime
        n_batch = 1  # Currently only works for 1 sequence at a time
        
        print("Building RR observation network...")
        
        # --------------------------------------------------------------------------------------------------------------
        # Shapes
        # --------------------------------------------------------------------------------------------------------------
        states_shape = [n_batch, None] + list(state_shape)
        state_shape_rr = [n_batch, 1] + list(state_shape[:-1]) + [1]
        actions_shape = [n_batch, None]
        action_shape = [n_batch, 1, n_actions]
        intgrd_batchsize = self.reward_redistribution_config['intgrd_batchsize']

        # --------------------------------------------------------------------------------------------------------------
        # Placeholders
        # --------------------------------------------------------------------------------------------------------------
        game_frames_placeholder = tf.placeholder(shape=states_shape, dtype=tf.uint8)
        game_actions_placeholder = tf.placeholder(shape=actions_shape, dtype=tf.int32)

        game_rewards_placeholder = tf.placeholder(shape=(None,), dtype=tf.float32)
        aux_target_placeholder = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        game_length_placeholder = tf.placeholder(shape=(), dtype=tf.int32)

        # --------------------------------------------------------------------------------------------------------------
        # Input to LSTM network
        # --------------------------------------------------------------------------------------------------------------
        # Create input layers (these will be set dynamically in LSTM loop)
        state_input_layer = RNNInputLayer(tf.zeros(state_shape_rr, dtype=tf.float32))
        statedelta_input_layer = RNNInputLayer(tf.zeros(state_shape_rr, dtype=tf.float32))
        rr_action_input_layer = RNNInputLayer(tf.zeros(action_shape, dtype=tf.float32))
        rr_time_input_layer = RNNInputLayer(ingametime.encode_value(tf.constant(0, shape=(n_batch,), dtype=tf.int32)))
        h_actor_input_layer = RNNInputLayer(tf.zeros(shape=policy_model.observation_features_shape))
        
        with tf.variable_scope(self.scopename, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("rr_visionsystem", reuse=tf.AUTO_REUSE):
                # RR observations will be single-/delta frames and features from A2C, actions, and ingame-time
                additional_inputs = [StopGradientLayer(h_actor_input_layer), rr_action_input_layer,
                                     ReshapeLayer(rr_time_input_layer, (n_batch, 1, ingametime.n_nodes_python))]
                lstm_prepoc_layers, _ = observation_network(single_frame=state_input_layer,
                                                            delta_frame=statedelta_input_layer,
                                                            additional_inputs=additional_inputs,
                                                            observation_network_config=self.observation_network_config)
                rr_input = lstm_prepoc_layers[-1]
                rr_input_layer = RNNInputLayer(rr_input)
            
            # ----------------------------------------------------------------------------------------------------------
            # LSTM network
            # ----------------------------------------------------------------------------------------------------------
            print("Building RR LSTM network...")
            
            # 1 node for predicting the return at the last timestep and 3 for auxiliary tasks
            n_rr_output_units = 1 + 3

            #
            # Initialization/activation functions
            #
            layerspecs = self.lstm_network_config['layers']
            init_specs = self.lstm_network_config['initializations']

            lstm_layer_index = [i for i, l in enumerate(layerspecs) if l['type'] == 'LSTMLayer'][0]
            lstm_specs = layerspecs[lstm_layer_index]
            n_lstm = lstm_specs['n_units']
            
            lstm_w_init = lambda scale: lambda *args, **kwargs: tf.truncated_normal(*args, **kwargs) * scale
            truncated_normal_init = lambda mean, stddev: \
                lambda *args, **kwargs: tf.truncated_normal(mean=mean, stddev=stddev, *args, **kwargs)

            if lstm_specs['a_out'] == 'linear':
                lstm_a_out = tf.identity
            elif lstm_specs['a_out'] == 'tanh':
                lstm_a_out = tf.tanh
            og_bias = truncated_normal_init(mean=init_specs['og_bias'], stddev=0.1)
            ig_bias = truncated_normal_init(mean=init_specs['ig_bias'], stddev=0.1)
            ci_bias = truncated_normal_init(mean=init_specs['ci_bias'], stddev=0.1)
            fg_bias = truncated_normal_init(mean=init_specs['fg_bias'], stddev=0.1)
            
            #
            # Layers setup
            #
            print("\tLSTM network for RR...")
            with tf.variable_scope('lstmnet', reuse=tf.AUTO_REUSE):
                
                # Store all layers in a list
                rr_layers = [rr_input_layer]
                
                #
                # Create layers before LSTM
                #
                rr_layers += layers_from_specs(incoming=rr_input_layer, layerspecs=layerspecs[:lstm_layer_index])
                rr_lstm_input_layer = rr_layers[-1]

                #
                # Create LSTM layer
                #
                w_ig = [lstm_w_init(init_specs['w_ig'][0]), lstm_w_init(init_specs['w_ig'][1])]
                w_og = [lstm_w_init(init_specs['w_og'][0]), lstm_w_init(init_specs['w_og'][1])]
                w_ci = [lstm_w_init(init_specs['w_ci'][0]), lstm_w_init(init_specs['w_ci'][1])]
                w_fg = [lstm_w_init(init_specs['w_fg'][0]), lstm_w_init(init_specs['w_fg'][1])]
                
                rr_lstm_layer = LSTMLayerGetNetInput(incoming=rr_lstm_input_layer, n_units=n_lstm,
                                                     name='LSTMCreditAssignment',
                                                     W_ci=w_ci, W_ig=w_ig, W_og=w_og, W_fg=w_fg,
                                                     b_ci=ci_bias([n_lstm]), b_ig=ig_bias([n_lstm]),
                                                     b_og=og_bias([n_lstm]), b_fg=fg_bias([n_lstm]),
                                                     a_ci=tf.tanh, a_ig=tf.sigmoid, a_og=tf.sigmoid, a_fg=tf.sigmoid,
                                                     a_out=lstm_a_out,
                                                     c_init=tf.zeros, h_init=tf.zeros, forgetgate=True,
                                                     precomp_fwds=False, store_states=True, return_states=False)
                rr_layers.append(rr_lstm_layer)
    
                #
                # Create layers after LSTM
                #
                if lstm_layer_index + 1 < len(layerspecs):
                    rr_layers += layers_from_specs(incoming=rr_layers[-1], layerspecs=layerspecs[lstm_layer_index + 1:])

                #
                # Create output layer
                #
                rr_layers.append(DenseLayer(incoming=rr_layers[-1], n_units=n_rr_output_units, a=tf.identity,
                                            W=lstm_w_init(1), b=constant([n_rr_output_units], 0.),
                                            name='DenseCAout'))
                rr_output_layer = rr_layers[-1]
                
            #
            # LSTM for integrated gradients
            # batched input; reduced LSTM layer for faster integrated gradient computation (we store input-activations
            # and don't have to recompute convolutions etc.);
            #
            with tf.variable_scope('lstmnet', reuse=tf.AUTO_REUSE):
                intgrd_layers = []
                
                # Placeholder for precomputed inputs to LSTM
                intgrd_input_shape = ([intgrd_batchsize, None] + rr_lstm_layer.cur_net_fwd.shape.as_list()[1:])
                intgrd_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[1] + intgrd_input_shape[1:])
                intgrd_input_layer = RNNInputLayer(tf.zeros((intgrd_input_shape[0], 1, intgrd_input_shape[2])))
                
                #
                # Create part of LSTM layer
                #
                
                # Reuse weights, activations and such from trained LSTM layer
                w_ig = rr_lstm_layer.W_bwd['ig']
                w_og = rr_lstm_layer.W_bwd['og']
                w_ci = rr_lstm_layer.W_bwd['ci']
                w_fg = rr_lstm_layer.W_bwd['fg']
                
                intgrd_lstm_layer = LSTMLayerSetNetInput(incoming=intgrd_input_layer, n_units=rr_lstm_layer.n_units,
                                                         name=rr_lstm_layer.name,
                                                         W_ci=w_ci, W_ig=w_ig, W_og=w_og, W_fg=w_fg,
                                                         b_ci=rr_lstm_layer.b['ci'], b_ig=rr_lstm_layer.b['ig'],
                                                         b_og=rr_lstm_layer.b['og'], b_fg=rr_lstm_layer.b['fg'],
                                                         a_ci=rr_lstm_layer.a['ci'], a_ig=rr_lstm_layer.a['ig'],
                                                         a_og=rr_lstm_layer.a['og'], a_fg=rr_lstm_layer.a['fg'],
                                                         a_out=rr_lstm_layer.a['out'],
                                                         c_init=tf.zeros, h_init=tf.zeros, forgetgate=True,
                                                         precomp_fwds=False, store_states=True, return_states=False)
                intgrd_layers.append(intgrd_lstm_layer)
    
                #
                # Create layers after LSTM
                #
                if lstm_layer_index + 1 < len(layerspecs):
                    intgrd_layers += layers_from_specs(incoming=intgrd_layers[-1],
                                                       layerspecs=layerspecs[lstm_layer_index + 1:])
        
                intgrd_layers.append(DenseLayer(incoming=intgrd_layers[-1], n_units=n_rr_output_units, a=tf.identity,
                                                W=lstm_w_init(1), b=constant([n_rr_output_units], 0.),
                                                name='DenseCAout'))
                intgrd_output_layer = intgrd_layers[-1]
        
        # --------------------------------------------------------------------------------------------------------------
        #  LSTM network loop
        # --------------------------------------------------------------------------------------------------------------
        #
        # Layers that require sequential computation (i.e. after LSTM incl. LSTM) will be computed in a tf.while loop
        #
        
        print("\tSetting up LSTM loop...")
        g = tf.get_default_graph()
        
        #
        # Get layers that require sequential computation (i.e. after LSTM incl. LSTM but excluding output layer)
        #
        rr_lstm_layer_position = [i for i, l in enumerate(rr_layers)
                                  if isinstance(l, LSTMLayer) or isinstance(l, LSTMLayerGetNetInput)][0]
        rr_layers_head = rr_layers[rr_lstm_layer_position + 1:-1]
        
        n_timesteps = game_length_placeholder - 1
        
        with tf.name_scope("RNNLoopLSTM"):
            #
            # Ending condition
            #
            def cond(time, *args):
                """Break if game is over by looking at n_timesteps"""
                return ~tf.greater(time, n_timesteps)
            
            #
            # Loop body
            #
            # Create initial tensors
            init_tensors = OrderedDict([
                ('time', tf.constant(0, dtype=tf.int32)),
                ('rr_net_fwd', tf.zeros([n_batch, 1] + rr_lstm_layer.cur_net_fwd.shape.as_list()[1:],
                                        dtype=tf.float32)),
                ('rr_lstm_internals', tf.expand_dims(tf.stack([rr_lstm_layer.c[-1], rr_lstm_layer.c[-1],
                                                               rr_lstm_layer.c[-1], rr_lstm_layer.c[-1],
                                                               rr_lstm_layer.c[-1]], axis=-1), axis=1)),
                ('rr_lstm_h', tf.expand_dims(rr_lstm_layer.h[-1], axis=1)),
                ('rr_pred_reward', tf.zeros([s if s >= 0 else 1 for s in rr_output_layer.get_output_shape()]))
            ])
            if len(rr_layers_head) > 0:
                init_tensors.update(OrderedDict(
                    [('dense_layer_{}'.format(i),
                      tf.zeros([s for s in l.get_output_shape() if s >= 0], dtype=tf.float32))
                     for i, l in enumerate(rr_layers_head)]))
            
            # Get initial tensor shapes in tf format
            init_shapes = OrderedDict([
                ('time', init_tensors['time'].get_shape()),
                ('rr_net_fwd', tensor_shape_with_flexible_dim(init_tensors['rr_net_fwd'], dim=1)),
                ('rr_lstm_internals', tensor_shape_with_flexible_dim(init_tensors['rr_lstm_internals'], dim=1)),
                ('rr_lstm_h', tensor_shape_with_flexible_dim(init_tensors['rr_lstm_h'], dim=1)),
                ('rr_pred_reward', tensor_shape_with_flexible_dim(init_tensors['rr_pred_reward'], dim=1)),
            ])
            if len(rr_layers_head) > 0:
                init_shapes.update(OrderedDict(
                    [('dense_layer_{}'.format(i), init_tensors['dense_layer_{}'.format(i)].get_shape())
                     for i, l in enumerate(rr_layers_head)]))

            def body_rr(time, rr_net_fwd, rr_lstm_internals, rr_lstm_h, rr_pred_reward, *args):
                """Loop over frames and additional inputs, compute network outputs and store hidden states and
                activations for debugging/plotting and integrated gradients calculation"""
                if self.lstm_network_config['reversed']:
                    time_index = n_timesteps - time
                else:
                    time_index = time
                
                #
                # Set state and state-deltas as network input
                #
                if self.observation_network_config['show_states']:
                    state_input_layer.update(tf.cast(tf.expand_dims(game_frames_placeholder[:, time_index, ..., -1:],
                                                                    axis=1), dtype=tf.float32))
                if self.observation_network_config['show_statedeltas']:
                    # Set the delta at timestep 0 to 0
                    delta_state = tf.cond(tf.equal(time_index, tf.constant(0, dtype=tf.int32)),
                                          lambda: tf.zeros_like(tf.cast(game_frames_placeholder[:, 0, ..., -1:],
                                                                        dtype=tf.float32)),
                                          lambda: (tf.cast(game_frames_placeholder[:, time_index, ..., -1:],
                                                           dtype=tf.float32)
                                                   - tf.cast(game_frames_placeholder[:, time_index - 1, ..., -1:],
                                                             dtype=tf.float32)))
                    statedelta_input_layer.update(tf.expand_dims(delta_state, axis=1))
                
                #
                # Set policy model input
                #
                h_actor_input_layer.update(policy_model.get_observation_features(
                        frame=tf.cast(tf.expand_dims(game_frames_placeholder[:, time_index], axis=1), dtype=tf.float32),
                        delta=statedelta_input_layer.out))
                
                #
                # Set time and actions as network input
                #
                rr_time_input_layer.update(self.ingametime.encode_value(time_index))
                if self.lstm_network_config['show_actions']:
                    curr_action = tf.expand_dims(game_actions_placeholder[:, time_index], axis=1)
                    curr_action = tf.cast(tf.one_hot(curr_action, depth=n_actions, axis=-1), dtype=tf.float32)
                    rr_action_input_layer.update(curr_action)
                
                #
                # Update and compute LSTM inputs
                #
                curr_rr_input = rr_input.get_output()
                rr_input_layer.update(curr_rr_input)
                
                #
                # Update LSTM cell-state and output with states from last timestep
                #
                rr_lstm_layer.c[-1], rr_lstm_layer.h[-1] = rr_lstm_internals[:, -1, :, -1], rr_lstm_h[:, -1, :]
                
                #
                # Calculate reward redistribution network output and append it to last timestep
                #
                rr_pred_reward = tf.concat([rr_pred_reward,
                                            rr_output_layer.get_output(prev_layers=[rr_input_layer])], axis=1)
                
                #
                # Store LSTM states for all timesteps for visualization
                #
                rr_lstm_internals = tf.concat([rr_lstm_internals,
                                               tf.expand_dims(
                                                   tf.stack([rr_lstm_layer.ig[-1], rr_lstm_layer.og[-1],
                                                             rr_lstm_layer.ci[-1], rr_lstm_layer.fg[-1],
                                                             rr_lstm_layer.c[-1]], axis=-1),
                                                   axis=1)],
                                              axis=1)
                
                #
                # Store LSTM output and forward-part of input activation for integrated gradients
                #
                rr_lstm_h = tf.concat([rr_lstm_h, tf.expand_dims(rr_lstm_layer.h[-1], axis=1)], axis=1)
                rr_net_fwd = tf.concat([rr_net_fwd, tf.expand_dims(rr_lstm_layer.cur_net_fwd, axis=1)], axis=1)
                
                #
                # Store output of optional layers above LSTM for debugging
                #
                rr_layers_head_activations = [l.out for l in rr_layers_head]
                
                #
                # Increment time
                #
                time += tf.constant(1, dtype=tf.int32)
                
                return [time, rr_net_fwd, rr_lstm_internals, rr_lstm_h, rr_pred_reward, *rr_layers_head_activations]
            
            wl_ret = tf.while_loop(cond=cond, body=body_rr, loop_vars=tuple(init_tensors.values()),
                                   shape_invariants=tuple(init_shapes.values()),
                                   parallel_iterations=self.loop_parallel_iterations, back_prop=True, swap_memory=True)
            
            # Re-Associate returned tensors with keys
            rr_returns = OrderedDict(zip(init_tensors.keys(), wl_ret))
            
            # Remove initialization timestep
            rr_returns['rr_net_fwd'] = rr_returns['rr_net_fwd'][:, 1:]
            rr_returns['rr_lstm_internals'] = rr_returns['rr_lstm_internals'][:, 1:]
            rr_returns['rr_lstm_h'] = rr_returns['rr_lstm_h'][:, 1:]
            rr_returns['rr_pred_reward'] = rr_returns['rr_pred_reward'][:, 1:]
            
            if len(rr_layers_head) > 0:
                for i, l in enumerate(rr_layers_head):
                    rr_returns['dense_layer_{}'.format(i)] = rr_returns['dense_layer_{}'.format(i)][:, 1:]
        
        accumulated_reward = tf.reduce_sum(game_rewards_placeholder)
        predicted_reward = rr_returns['rr_pred_reward']
        
        #
        # Track exponential mean of reward
        #
        with tf.variable_scope(self.scopename, reuse=tf.AUTO_REUSE):
            emean_reward = tf.get_variable('emean_reward', initializer=tf.constant(0., dtype=tf.float32),
                                           trainable=False)
        d = 0.99
        emean_reward = tf.assign(emean_reward, d * emean_reward + (1. - d) * accumulated_reward)
        
        #
        # Error for reward prediction
        #
        with g.control_dependencies([emean_reward]):
            reward_prediction_error = predicted_reward[0, -1, 0] - accumulated_reward
        
        # --------------------------------------------------------------------------------------------------------------
        #  Loss and Update Steps for reward redistribution network
        # --------------------------------------------------------------------------------------------------------------
        
        print("\tSetting up RR updates...")
        layers_to_train = rr_layers + lstm_prepoc_layers
        rr_trainables = [t for t in tf.trainable_variables() if t.name.find('RR/lstmnet') != -1]
        rr_trainables_vs = [t for t in tf.trainable_variables() if t.name.find('RR/rr_visionsystem') != -1]
        rr_trainables = rr_trainables + rr_trainables_vs
        
        #
        # RR Update
        #
        
        # Main loss target
        rr_loss_last_timestep = tf.square(reward_prediction_error)
        
        # Add regularization penalty
        rr_reg_penalty = regularize(layers=layers_to_train, l1=self.training_config['l1'],
                                    l2=self.training_config['l2'], regularize_weights=True, regularize_biases=True)
        rr_loss = rr_loss_last_timestep + rr_reg_penalty
        
        # Auxiliary losses
        cumsum_rewards = tf.cumsum(game_rewards_placeholder)
        target_all_ts = tf.constant(0., dtype=tf.float32)  # avoid errors if aux. losses aren't used
        if self.reward_redistribution_config['cont_pred_w']:
            # Calculate mean over aux. losses and add them to main loss
            target_all_ts = aux_target_placeholder
            rr_loss_all_timesteps = (tf.reduce_sum(tf.reduce_mean(tf.square(aux_target_placeholder
                                                                            - predicted_reward[0, :, 1:-1]), axis=0))
                                     + tf.reduce_mean(tf.square(cumsum_rewards - predicted_reward[0, :, -1]))) / 3.
            rr_loss += rr_loss_all_timesteps * self.reward_redistribution_config['cont_pred_w']
        
        # Get gradients
        rr_grads = tf.gradients(rr_loss, rr_trainables)
        
        if self.training_config['clip_gradients']:
            rr_grads, _ = tf.clip_by_global_norm(rr_grads, self.training_config['clip_gradients'])
        
        # Set up optimizer
        rr_update = tf.constant(0)
        if self.training_config['optimizer_params']['learning_rate'] != 0:
            with tf.variable_scope('rr_update', tf.AUTO_REUSE):
                optimizer = getattr(tf.train,
                                    self.training_config['optimizer'])(**self.training_config['optimizer_params'])
                rr_update = optimizer.apply_gradients(zip(rr_grads, rr_trainables))
        
        # --------------------------------------------------------------------------------------------------------------
        #  Integrated Gradients
        # --------------------------------------------------------------------------------------------------------------

        print("\tSetting up Integrated Gradients...")
        # Create input that interpolates between zeroed- and full input sequence
        intgrd_w_placeholder = tf.placeholder(dtype=tf.float32, shape=(intgrd_batchsize,))
        intgrd_input = tf.concat(
                [intgrd_input_placeholder * intgrd_w_placeholder[s] for s in range(intgrd_batchsize)],
                axis=0)

        #
        # Ending condition
        #
        def cond(time, *args):
            """Break if game is over"""
            return ~tf.greater(time, n_timesteps)

        #
        # Loop body
        #
        # Create initial tensors
        init_tensors = OrderedDict([
            ('time', tf.constant(0, dtype=tf.int32)),
            ('lstm_c', tf.zeros((intgrd_batchsize, n_lstm), dtype=tf.float32)),
            ('lstm_h', tf.zeros((intgrd_batchsize, n_lstm), dtype=tf.float32)),
            ('pred', tf.zeros(intgrd_output_layer.get_output_shape()))
        ])

        # Get initial tensor shapes in tf format
        init_shapes = OrderedDict([
            ('time', init_tensors['time'].get_shape()),
            ('lstm_c', init_tensors['lstm_c'].get_shape()),
            ('lstm_h', init_tensors['lstm_h'].get_shape()),
            ('pred', init_tensors['pred'].get_shape()),
        ])

        def body_intgrd(time, lstm_c, lstm_h, pred):
            """Loop body for integrated gradients"""
            if self.lstm_network_config['reversed']:
                time_index = n_timesteps - time
            else:
                time_index = time

            #
            # Update layer with precomputed forward input activations for LSTM
            #
            intgrd_input_layer.update(intgrd_input[:, time_index:time_index + 1, :])

            #
            # Update LSTM states
            #
            intgrd_lstm_layer.c[-1], intgrd_lstm_layer.h[-1] = lstm_c, lstm_h

            #
            # Calculate output
            #
            pred = intgrd_output_layer.get_output()

            lstm_c = intgrd_lstm_layer.c[-1]
            lstm_h = intgrd_lstm_layer.h[-1]

            # Increment time
            time += tf.constant(1, dtype=tf.int32)

            return [time, lstm_c, lstm_h, pred]

        wl_ret = tf.while_loop(cond=cond, body=body_intgrd, loop_vars=tuple(init_tensors.values()),
                               shape_invariants=tuple(init_shapes.values()),
                               parallel_iterations=self.loop_parallel_iterations, back_prop=True, swap_memory=True)

        intgrd_pred = wl_ret[-1]
        # For reward redistribution, use only main task and aux. task for accumulated reward prediction
        intgrd_pred = intgrd_pred[..., 0] + intgrd_pred[..., -1]

        # Get gradients, set NaNs to 0
        grads = tf.gradients(intgrd_pred, intgrd_input)[0]
        grads = tf.where(tf.is_nan(grads), tf.zeros_like(grads), grads)
        # Get prediction at first sample as we need zero-sample prediction for quality check of integrated gradients
        intgrd_pred_first_sample = intgrd_pred[0]
        # Calc gradients, sum over batch dimension
        intgrd_grads = tf.reduce_sum(grads, axis=0)
        # Scale by original input
        intgrd_grads *= intgrd_input_placeholder[0]
        # Sum over features=lstm units
        intgrd_grads = tf.reduce_sum(intgrd_grads, axis=-1)
        
        # --------------------------------------------------------------------------------------------------------------
        #  TF-summaries
        # --------------------------------------------------------------------------------------------------------------
        tf.summary.scalar("Environment/reward", accumulated_reward)
        tf.summary.scalar("Environment/emean_reward", emean_reward)
        tf.summary.scalar("Environment/n_timesteps", n_timesteps)
        
        tf.summary.scalar("RR/rr_loss_last_timestep", rr_loss_last_timestep)
        if self.reward_redistribution_config['cont_pred_w']:
            tf.summary.scalar("RR/rr_loss_all_timesteps", rr_loss_all_timesteps)
        tf.summary.scalar("RR/rr_reg_penalty", rr_reg_penalty)
        tf.summary.scalar("RR/rr_loss", rr_loss)
        tf.summary.scalar("RR/predicted_reward", predicted_reward[0, -1, 0])
        
        if self.write_histograms:
            [tf.summary.histogram("activations/RR/{}".format(n), values=rr_returns['rr_lstm_internals'][0, -1, 0, i])
             for i, n in enumerate(['rr_lstm_ig', 'rr_lstm_og', 'rr_lstm_ci', 'rr_lstm_fg'])]

            tf.summary.histogram("activations/RR/lstm_h", rr_returns['rr_lstm_h'][0, -1, :])

            [tf.summary.histogram("gradients/RR/{}".format(t.name), values=g) for g, t in zip(rr_grads, rr_trainables)]
            [tf.summary.histogram("weights/RR/{}".format(t.name), values=t) for t in rr_trainables]
        
        # --------------------------------------------------------------------------------------------------------------
        #  Publish
        # --------------------------------------------------------------------------------------------------------------
        
        # Placeholders
        placeholders = OrderedDict(
            game_frames_placeholder=game_frames_placeholder,
            game_actions_placeholder=game_actions_placeholder,
            game_rewards_placeholder=game_rewards_placeholder,
            game_length_placeholder=game_length_placeholder,
            aux_target_placeholder=aux_target_placeholder,
            intgrd_input_placeholder=intgrd_input_placeholder,
            intgrd_w_placeholder=intgrd_w_placeholder
        )
        
        # Data
        data_tensors = OrderedDict(
                lstm_internals=rr_returns['rr_lstm_internals'][0],
                lstm_h=rr_returns['rr_lstm_h'],
                intgrd_gradients=intgrd_grads,
                intgrd_lstm_net_fwd=rr_returns['rr_net_fwd'],
                intgrd_pred_last_sample=intgrd_pred[-1],
                intgrd_pred_first_sample=intgrd_pred_first_sample,
                rr_loss_last_timestep=rr_loss_last_timestep,
                rr_loss=rr_loss,
                predictions=predicted_reward[0, :, :],
                aux_target_all_ts=target_all_ts,
                pred_return=predicted_reward[0, -1, 0]
        )
        
        # Operations
        operation_tensors = OrderedDict(
            rr_update=rr_update
        )
        
        # Summaries
        summaries = OrderedDict(
            all_summaries=tf.summary.merge_all()
        )
        
        self.placeholders = placeholders
        self.data_tensors = data_tensors
        self.operation_tensors = operation_tensors
        self.summaries = summaries
        
    def reward_redistribution(self, tf_session, states, actions, rewards, aux_target, avg_reward,
                              redistribute_reward=False, details=False, use_reward_redistribution=True,
                              summaries=False, update=True, verbose=True):
        """Perform reward redistribution without junking
        
        batchsize != 1 currently not supported; see reward_redistribution_junked() for more information;
        """
        
        if verbose:
            print(" started loop with {} timesteps...".format(len(rewards)), end="")
            starttime = time.time()
            sys.stdout.flush()
        
        #
        # Set placeholder values
        #
        placeholder_values = OrderedDict(
            game_frames_placeholder=states,
            game_actions_placeholder=actions,
            game_rewards_placeholder=rewards,
            game_length_placeholder=len(rewards),
            aux_target_placeholder=aux_target
        )
        feed_dict = dict(((self.placeholders[k], placeholder_values[k]) for k in placeholder_values.keys()))
        
        #
        # Decide which tensors to compute
        #
        data_keys = ['rr_loss', 'rr_loss_last_timestep', 'pred_return']
        if redistribute_reward:
            data_keys += ['intgrd_lstm_net_fwd', 'predictions']
        if details:
            data_keys += ['lstm_internals', 'lstm_h', 'predictions', 'aux_target_all_ts']
        
        data_tensors = [self.data_tensors[k] for k in data_keys]
        
        operation_keys = []
        if update:
            operation_keys += ['rr_update']
        operation_tensors = [self.operation_tensors[k] for k in operation_keys]
        
        summary_keys = []
        if summaries:
            summary_keys += ['all_summaries']
        summary_tensors = [self.summaries[k] for k in summary_keys]

        #
        # Run graph and re-associate return values with keys in dictionary
        #
        ret = tf_session.run(data_tensors + summary_tensors + operation_tensors, feed_dict)
        
        ret_dict = OrderedDict(((k, ret[i]) for i, k in enumerate(data_keys)))
        del ret[:len(data_keys)]
        ret_dict.update(OrderedDict(((k, ret[i]) for i, k in enumerate(summary_keys))))
        
        #
        # Check reward redistribution and integrated gradients quality
        #
        ret_dict['rel_error'] = 1
        ret_dict['rr_quality'] = 0
        if redistribute_reward:
            #
            # Calculate squared percentage prediction error to scale reward mixing (ignore redistribution if error>20%)
            #
            use_reward_redistribution_quality_threshold = \
                self.reward_redistribution_config['use_reward_redistribution_quality_threshold']
            target = np.sum(rewards)
            epsilon_sqd = np.sqrt(np.clip(np.abs(avg_reward), a_min=1e-5, a_max=None))
            prediction = ret_dict['pred_return']
            sqd_perd_pred_err = ((target - prediction) ** 2) / (target**2 + epsilon_sqd)
            if verbose:
                print("\tsqd_perd_pred_err: {} (t:{}, p:{}".format(sqd_perd_pred_err, target, prediction))
            ret_dict['rel_error'] = sqd_perd_pred_err
            
            # Don't compute integrated gradients if error is too high
            if sqd_perd_pred_err < use_reward_redistribution_quality_threshold:
                intgrds, intgrdperc, zero_sample_pred = self.integrated_gradients(
                        tf_session=tf_session, lstm_inputs=ret_dict['intgrd_lstm_net_fwd'],
                        game_len=len(rewards), intgrd_steps=self.reward_redistribution_config['intgrd_steps'],
                        intgrd_batchsize=self.reward_redistribution_config['intgrd_batchsize'], verbose=verbose)
                
                # Quality of reward redistribution is only > 0 if integrated gradients is good enough
                ret_dict['rr_quality'] = 1 - sqd_perd_pred_err
                
                # Check if integrated gradients signal is within +/- 20% error range
                if (80. <= intgrdperc <= 120.) and use_reward_redistribution:
                    # Correct for integrated gradients error
                    intgrds += zero_sample_pred / len(rewards)
                    intgrdssum = np.sum(intgrds)
                    error = prediction - intgrdssum
                    intgrds += error / len(rewards)
                    
                    # Correct for return prediction error
                    intgrds[:] *= np.clip(rewards.sum() / (prediction +
                                                           (np.sign(prediction) * (np.sqrt(epsilon_sqd) / 5))),
                                          a_min=1e-5, a_max=1.5)
                    
                    ret_dict['redistributed_reward'] = intgrds
                    ret_dict['intgrd_from_lstm'] = intgrds
                else:
                    ret_dict['redistributed_reward'] = rewards
                    ret_dict['intgrd_from_lstm'] = intgrds
            else:
                ret_dict['redistributed_reward'] = rewards
                ret_dict['intgrd_from_lstm'] = np.zeros_like(rewards)
        else:
            ret_dict['redistributed_reward'] = rewards
            ret_dict['intgrd_from_lstm'] = np.zeros_like(rewards)
        
        if verbose:
            print("done! ({}sec)".format(time.time() - starttime))
        
        return ret_dict
    
    def reward_redistribution_junked(self, tf_session, states, actions, rewards, avg_reward, redistribute_reward=True,
                                     use_reward_redistribution=True, update=True, details=False, summaries=False,
                                     junksize=500, verbose=True):
        """Perform reward redistribution on longer sequences junk-wise
        
        batchsize != 1 currently not supported; Variable sizes for n_timesteps may be used;
        
        Parameters
        ----------
        tf_session : tensorflow session
            tensorflow session to compute the graph in
        states : numpy array
            Game frames of shape (batchsize, n_timesteps, x, y, c)
        actions : numpy array
            Taken actions of shape (batchsize, n_timesteps, 1)
        rewards : numpy array
            Reward from environment of shape (n_timesteps,)
        avg_reward : float
            Average reward used to compute reward redistribution quality measure
        redistribute_reward : bool
            Compute reward redistribution?
        use_reward_redistribution : bool
            Use reward redistribution?
        update : bool
            Update weights of reward redistribution model?
        details : bool
            Enable computation and logging of debugging details?
        summaries : bool
            Enable computation of tensorboard summaries?
        junksize : int
            Number of timesteps per sequence junk
        verbose : bool
            Enable printing to console?
            
        Returns
        ----------
        dict
            Dictionary containing evaluated tensors; Non-optional key/values are:
            
            ----------
            'redistributed_reward' : numpy array
                Final redistributed reward in array of shape (n_timesteps,)
            'intgrd_from_lstm' : numpy array
                Integrated gradients signal of shape (n_timesteps,)
            'rr_loss' : float
                Mean loss for all targtes of reward redistribution model over all junks
            'rr_loss_last_timestep' : float
                Mean loss for main target (=return prediction) of reward redistribution model over all junks
            'rel_error' : float
                Mean relative error of reward redistribution model over all junks
            'rr_quality' : float
                Quality measure of reward redistribution model for last junk
        """
        seq_len = len(rewards)
        n_junks = int(np.ceil(seq_len / junksize))
        
        # Overlap junks
        n_junks += n_junks - 1
        halved_junksize = int(junksize / 2)
        
        redistributed_reward = np.copy(rewards)
        intgrd_from_lstm = np.zeros_like(rewards)
        rr_quality = np.empty_like(rewards)
        
        # Prepare auxiliary tasks
        aux_target = np.empty((rewards.shape[0], 2))
        aux_target_temp = np.concatenate([rewards, self.aux_target_pad], axis=0)
        aux_target[:, 0] = np.convolve(aux_target_temp, self.aux_target_filter, 'valid')
        aux_target[:, 1] = np.sum(rewards) - np.cumsum(rewards)
        
        if verbose:
            print(" started loop with {} steps in {} junks...".format(len(rewards), n_junks))
            starttime = time.time()
        
        #
        # Loop over junks and redistribute reward; Only store first half of overlapping junks and overwrite rest;
        #
        junk_ret_dicts = []
        for junk in range(n_junks):
            junkslice = slice(junk*halved_junksize, junk*halved_junksize+junksize)
            junk_ret_dict = self.reward_redistribution(tf_session,
                                                       states[:, junkslice], actions[:, junkslice],
                                                       rewards[junkslice],
                                                       aux_target[junkslice], avg_reward=avg_reward,
                                                       redistribute_reward=redistribute_reward, details=details,
                                                       use_reward_redistribution=use_reward_redistribution,
                                                       summaries=summaries, update=update, verbose=False)
            redistributed_reward[junkslice] = junk_ret_dict['redistributed_reward']
            intgrd_from_lstm[junkslice] = junk_ret_dict['intgrd_from_lstm']
            rr_quality[junkslice] = junk_ret_dict['rr_quality']
            junk_ret_dicts.append(junk_ret_dict)
        
        #
        # If multiple junks were used, concatenate sequences for plotting etc. accordingly
        #
        if n_junks > 1:
            ret_dict = dict(((k, np.concatenate([jrd[k][:halved_junksize] for jrd in junk_ret_dicts], axis=0))
                             if k == 'lstm_internals' or k == 'predictions' or k == 'aux_target_all_ts'
                             else (k, np.concatenate([jrd[k][:, :halved_junksize] for jrd in junk_ret_dicts], axis=1))
                             if k == 'lstm_h'
                             else (k, junk_ret_dicts[-1][k])
                             if k == 'all_summaries'
                             else (k, None)
                             if (k == 'intgrd_lstm_net_fwd' or k == 'redistributed_reward'
                                 or k == 'intgrd_from_lstm' or k == 'rr_quality')
                             else (k, [jrd[k] for jrd in junk_ret_dicts])
                             for k in junk_ret_dicts[0].keys()))
        else:
            ret_dict = junk_ret_dicts[0]
        
        #
        # Apply eligibility trace to redistributed reward
        #
        et_redistributed_reward = np.zeros_like(redistributed_reward)
        et_redistributed_reward[-1] = redistributed_reward[-1]
        for t in reversed(range(0, len(redistributed_reward) - 1)):
            et_redistributed_reward[t] = self.reward_redistribution_config['lambda_eligibility_trace'] * \
                                         et_redistributed_reward[t + 1] + redistributed_reward[t]
        
        #
        # Add mandatory fields to return dictionary
        #
        ret_dict['redistributed_reward'] = et_redistributed_reward
        ret_dict['intgrd_from_lstm'] = intgrd_from_lstm
        ret_dict['rr_loss'] = np.mean(ret_dict['rr_loss'])
        ret_dict['rr_loss_last_timestep'] = np.mean(ret_dict['rr_loss_last_timestep'])
        ret_dict['rel_error'] = np.mean(ret_dict['rel_error'])
        # rr_quality taken from the last junk. (final return)
        ret_dict['rr_quality'] = rr_quality
        
        if verbose:
            print("\t...done! ({}sec)".format(time.time() - starttime))
        
        return ret_dict
    
    def integrated_gradients(self, tf_session, lstm_inputs, game_len, intgrd_steps, intgrd_batchsize, verbose=True):
        """Compute integrated gradients
        
        batchsize != 1 currently not supported; Variable sizes for n_timesteps may be used; intgrd_steps must be
        dividable by intgrd_batchsize;
        
        Parameters
        ----------
        tf_session : tensorflow session
            tensorflow session to compute the graph in
        lstm_inputs : numpy array
            Pre-computed input activations for reward redistribution LSTM network of shape (batchsize, n_timesteps, f)
        game_len : int
            Number of timesteps in game
        intgrd_steps : int
            Number of steps/interpolations to use in integrated gradients
        intgrd_batchsize : int
            Batchsize to use when parallelizing integrated gradients computation
        verbose : bool
            Enable printing to console?
            
        Returns
        ----------
        numpy array
            Numpy array with integrated grdients signal of shape (n_timesteps,)
        float
            Quality measure for integrated gradients in percent (see RUDDER paper)
        float
            Reward redistribution model output for zeroed input sequence for rescaling of signal
        """
        if verbose:
            print(" started integrated gradients with {} intgrd-steps...".format(intgrd_steps), end="")
            starttime = time.time()
            sys.stdout.flush()
        
        # Create multiplier for interpolating between full and zeroed input sequence
        intgrd_w = np.linspace(0, 1, num=intgrd_steps, dtype=np.float32)
        
        intgrds = None
        zero_sample_pred = None
        full_sample_pred = None
        
        # Set up minibatches
        n_mbs = int(intgrd_steps / intgrd_batchsize)
        if (intgrd_steps % intgrd_batchsize) != 0:
            raise ValueError("intgrd stepsize not dividable by intgrd batchsize!")
        
        #
        # Loop over minibatches and compute integrated gradients
        #
        for mb in range(n_mbs):
            if verbose:
                print(".", end="")
            
            curr_intgrd_w = intgrd_w[mb * intgrd_batchsize:(mb + 1) * intgrd_batchsize]
            
            placeholder_values = OrderedDict(
                intgrd_input_placeholder=lstm_inputs,
                intgrd_w_placeholder=curr_intgrd_w,
                game_length_placeholder=game_len
            )
            feed_dict = dict(((self.placeholders[k], placeholder_values[k]) for k in placeholder_values.keys()))
            
            data_keys = ['intgrd_gradients']
            if mb == 0:
                # get prediction for 0-sample
                data_keys += ['intgrd_pred_first_sample']
            if mb == (n_mbs - 1):
                # get prediction for full sample
                data_keys += ['intgrd_pred_last_sample']
            data_tensors = [self.data_tensors[k] for k in data_keys]
        
            ret = tf_session.run(data_tensors, feed_dict)
            
            ret_dict = OrderedDict(((k, ret[i]) for i, k in enumerate(data_keys)))
            
            if intgrds is None:
                intgrds = ret_dict['intgrd_gradients']
            else:
                intgrds += ret_dict['intgrd_gradients']
                
            if mb == 0:
                zero_sample_pred = ret_dict['intgrd_pred_first_sample']
            if mb == (n_mbs - 1):
                full_sample_pred = ret_dict['intgrd_pred_last_sample']
        
        intgrds /= intgrd_steps
        
        #
        # Compute percentage of integrated gradients reconstruction quality
        #
        intgrdssum = intgrds.sum()
        diff = full_sample_pred - zero_sample_pred
        if diff != 0:
            intgrdperc = 100. / diff * intgrdssum
        else:
            # in case 0-sequence and full-sequence are equal, take a heuristic to decide on intgrdperc
            intgrdperc = 100. + intgrdssum
        
        # Integrated gradients shows instabilities at last timestep, so we set them to 0
        intgrds[-10:] = 0
        
        if verbose:
            print("pred", full_sample_pred, "0pred", zero_sample_pred, "diff", diff, "intgrd", intgrdssum,
                  "perc", intgrdperc)
            print("done! ({}sec)".format(time.time() - starttime))
        
        return intgrds, intgrdperc, zero_sample_pred
