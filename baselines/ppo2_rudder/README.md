# RUDDER PPO2

Uses RUDDER on PPO to solve environments with delayed rewards.

Please note that in this implementation the reward redistribution model is not optimized and computed sequentially for each game.

- Original paper: https://arxiv.org/abs/1806.07857
- Requires [Tensorflow Layer Library (TeLL)](https://github.com/bioinf-jku/tensorflow-layer-library) package (v1.0)
- `python3 -m baselines.ppo2_rudder.run_atari.py --config ppo2_rudder/configs/Venture.json` runs the algorithm with configuration file "ppo2_rudder/configs/Venture.json".

We use json configuration files to make adjustments to PPO2 agent and RUDDER easier.
Examples are provided in [baselines/ppo2_rudder/configs](baselines/ppo2_rudder/configs). Available settings are:


      specs : str
        Name of the current run
      cuda_gpu : int
        GPU to use (-1 for CPU-only)
      tensorflow_allow_growth : bool
          Allow tensorflow to dynamically allocate memory? (if not, the full memory will be allocated)
      random_seed : int
        Random seed to use for tensorflow and numpy
      max_n_frames : int
        Max. number of frames to have per game
      policy : "cnn", "lstm", or "lstmdense"
        Architecture to use for PPO2 (see baselines/ppo2_rudder/policies.py)
      working_dir : str
        Working directory path, e.g. "workingdir/ppo2_rudder"
      plot_at : int
        Create plots every plot_at updates
      save_at : int
        Create checkpoints after save_at PPO2 updates but only keep lates n_savefiles, besides permanent checkpoints 
        and checkpoints for best runs
      n_savefiles : int
        Max. number of checkpoints to keep (besides permanent checkpoints and best runs)
      load_file_dict : dict
        Dictionary for paths to load checkpoints from
        ----
        rr_buffer : str
            Lessons buffer save-file, e.g. "/FOLDER/saves/state-3700.h5py"
        states : str
            General training state save-file, e.g. "/FOLDER/saves/state-3700.pkl.zip",
        model : str
            Tensorflow variables for PPO2 model, e.g. "/FOLDER/saves/checkpoint-3700",
        RR : str
            Tensorflow variables for reward redistribution model, e.g. "/FOLDER/saves/checkpoint-3700"
      bl_config : dict
        Configuration concerning baselines package settings:
        ----
        env : str
            Name of environment, e.g. "VentureNoFrameskip-v4" for Venture with "NoFrameskip-v4" input preprocessing
        num_timesteps : int
            Number of frames to train
        episode_reward : bool
            Makes getting_reward == end-of-episode, but only reset on true game over.
            Done for games like Bowling, where you get reward after 2 shoots, but the game
            itself takes very long to finish. episode_reward/episode_life/episode_frame are mutually exclusive.
        episode_life : bool
            Standard deepmind environment for most games. episode_reward/episode_life/episode_frame are mutually 
            exclusive.
        episode_frame : bool
            Makes end-of-episode after 490 frames, but only reset on true game over.
            Done for games like Enduro, Dubledunk, Icehockey... where you can stop the game
            and continue as a new episode. episode_reward/episode_life/episode_frame are mutually exclusive.
        temperature_decay : bool
            Decay temperature of softmax to sample actions from with average game lenght?
        num_actors : int
            Number of parallel actors to use
        lr_coef : float
            Scaling factor for original baselines learning rate
        ent_coef : float
            Entropy coefficient for loss
      rudder_config : dict
        Configuration concerning RUDDER:
        ----
        write_histograms : false
            Write histogram of weights/activations to tensorboard? (expensive but good for debugging)
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
                Network config to preprocess frames, see example files
            prepoc_deltas : list of dicts
                Network config to preprocess frame deltas, see example files
            prepoc_observations : list of dicts
                Network config to preprocess features from frame and frame-delta preprocessing networks, see example 
                files
        lstm_network_config : dict
            Dictionary containing config for LSTM network:
            -----
            show_actions : bool
                Show taken actions to LSTM?
            reversed : bool
                Process game sequence in reversed order?
            layers : list of dicts
                Network config for LSTM network and optional additional dense layers, see example files
            initializations : dict
                Initialization config for LSTM network, see example files
            timestep_encoding : dict
                Set "max_value" and "triangle_span" for TeLL.utiltiy.misc_tensorflow.TriangularValueEncoding class
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
                

