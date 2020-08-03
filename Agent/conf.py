import torch
from common_utils import AttrDict
import gym


class AgentConf(AttrDict):
    def __init__(self, obs_space: gym.Space = None, action_space: gym.Space = None):
        super().__init__(self)
        self.obs_space, self.action_space = obs_space, action_space
        self.num_instances = 1
        self.replay_size = int(5e4)
        self.log_dir = "logs"

        self.fpp = torch.float32  # floating point precision
        self.inference_device = "cpu:0" if torch.cuda.is_available() else "cpu:0"
        self.training_device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        self.dump_period = 50  # dump things from trainer after this many steps

        self.mlp_hidden_dims = [256]
        self.mlp_activation = torch.nn.ReLU
        self.inference_input_keys = "obs", "idx"  # environment keys required to run inference
        self.algorithm = "random"  # [random | sac | ]

        # Optional Components
        self.use_double_q = True  # keep 2 espimates of Q function to address over-estimation bias [Mnih et al]
        self.use_soft_targets = True  # Target update method (see SAC paper)
        self.squash_rewards = True  # Reduce reward variance with transform from [arXiv:1805.11593]
        """^^^Instantiate a replay wrapper which reduces variance of the stored rewards for numeric stability"""

        self.use_sde = False  # State Dependent Exploration [arXiv:2005.05719]
        """Parametize a random network that predicts noise for our actions conditioned on state.
        The weights and activations are designed to ensure the output is distributed as: Normal(μ=0,σ=1)
        We use this for reparametization instead of a true random gaussian sample.
        This way the noise is state-dependent & temporally correlated; less jerky         
        """

        # hyper params
        self.lr = 3e-4  # learning rate
        self.gamma = 0.99  # discount factor
        self.tau = 5e-2  # soft target update rate
        self.batch_size = 256
        self.temporal_len = 50
        self.grad_clip = 20
        self.initial_log_alpha = -2  # starting point for entropy tuning (SAC)
        self.sde_update_interval = 50 # how many steps before resetting SDE gaussian

        self.enable_profiling = False

    @property
    def discrete(self):
        return isinstance(self.action_space, gym.spaces.Discrete)
