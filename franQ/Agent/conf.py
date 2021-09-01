from pathlib import Path
from franQ.common_utils import AttrDict
from torch import multiprocessing as mp


class AgentConf(AttrDict):
    def __init__(self):
        AttrDict.__init__(self)
        # I/O
        self.obs_space = None
        self.action_space = None
        self.discrete = None
        self.global_step = mp.Value("i", 0)
        self.num_instances = 2
        # environment keys required to run inference
        self.inference_input_keys = "obs_1d", "obs_2d", "idx", "achieved_goal", "desired_goal"

        # devices
        import torch
        self.training_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.inference_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.dtype = torch.float32
        # Logging
        self.log_dir = Path("logs")
        self.enable_timers = False
        self.log_interval = 50
        self.param_update_interval = 50

        # replay
        self.batch_size = 256
        self.replay_size = int(5e4)
        self.temporal_len = 50  # Note: this implicitly applies temoral consistency loss (Pohlen et al 2018)
        self.clip_grad_norm = 20

        # Algo and components
        self.use_squashed_rewards = True  # Apply pohlen transform [arXiv:1805.11593] to reduce variance and stabilize training
        self.use_hard_updates = False  # False-> Use polyak averaging for target networks. True-> Periodic hard updates
        self.use_target_actor = True
        self.use_target_encoder = False

        self.algorithm = "deep_q_learning"
        self.use_nStep_lowerbounds = True  # Lowerbound on Q to speed up convergence [https://arxiv.org/abs/1611.01606]
        self.nStep_return_steps = 1000
        self.use_max_entropy_q = True  # Intrinsic reward for random behavior while still following objective [https://arxiv.org/abs/1812.11103]
        self.use_HER = False
        self.her_mode = "final" # final | random


        # TODO: Work In Progress:
        self.use_distributional_sac = True
        self.use_bootstrap_minibatch_nstep = False
        self.use_async_train = True  # TODO: Fix pipeline stall issue when this is disabled
        # ^^^^^^^^^^^^^^^^^^^^^^ API V4 Components
        self.use_decoder = False  # Map latent space back to obs space (for visualization only)
        self.use_hsv_data_augmentation = False  # Use standard vision data augmentation tricks on hsv images
        self.use_strided_rnn = False

        # hyperparams
        self.init_log_alpha = -2  # starting point for entropy tuning (SAC)
        self.gamma = 0.99
        self.learning_rate = 3e-4
        self.tau = 5e-2
        self.hard_update_interval = 200
        self.n_quantiles = 32
        self.quantile_embedding_dim = 64
        self.kappa = 1.0

        # Model arch hyperparams
        self.conv_channels = 32
        self.conv_depth = 4
        self.enc1d_hidden_dims = [256]
        self.pi_hidden_dims = [256]
        self.critic_hidden_dims = [256, 256]
        self.num_critics = 2
        self.num_q_predictions = 10
        self.latent_state_dim = 256
        self.top_quantiles_to_drop = 0.2
