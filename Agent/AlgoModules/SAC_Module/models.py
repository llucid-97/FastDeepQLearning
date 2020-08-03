import torch
from torch import nn, Tensor as _Tensor
from Agent.Utils import mlp, gumbel_softmax, gaussian
from torch import nn, Tensor, distributions
from Agent.conf import AgentConf


def make_actor(conf: AgentConf, input_dim):
    from gym.spaces import Discrete
    if isinstance(conf.action_space, Discrete):
        return gumbel_softmax.GumbelMLP(input_dim, conf.action_space.n, conf.mlp_hidden_dims, conf.mlp_activation)
    else:
        return gaussian.GaussianMLP(input_dim, conf.action_space.shape[-1], conf.mlp_hidden_dims, conf.mlp_activation)


def make_critic(conf: AgentConf, input_dim):
    from gym.spaces import Discrete
    num_actions = conf.action_space.n if isinstance(conf.action_space, Discrete) else conf.action_space.shape[-1]
    if conf.use_double_q:
        return DoubleQNetwork(input_dim + num_actions, 1, conf.mlp_hidden_dims, conf.mlp_activation)
    else:
        return mlp.MLP(input_dim + num_actions, 1, conf.mlp_hidden_dims, conf.mlp_activation)

def make_sde_actor(conf: AgentConf, input_dim):
    from gym.spaces import Discrete
    if isinstance(conf.action_space, Discrete):
        raise NotImplementedError("TODO: Add this once gaussian is in working state and gumbel softmax policy has been tested in regular SAC!")
        return gumbel_softmax.GumbelMLP(input_dim, conf.action_space.n, conf.mlp_hidden_dims, conf.mlp_activation)
    else:
        return SDEGaussianPolicy(input_dim, conf.action_space.shape[-1], conf.mlp_hidden_dims, conf.mlp_activation)



class SDEGaussianPolicy(nn.Module):
    # use an mlp to predict mean and std of gaussian, and use reparametization trick to sample it
    def __init__(self, in_features, out_features, hidden_sizes: tuple, activation_class=nn.ReLU,
                 log_sig_min=-20.0, log_sig_max=2, epsilon=1e-4):
        super(SDEGaussianPolicy, self).__init__()

        # define bounds on variance for numeric stability
        self.log_sig_min, self.log_sig_max, self.epsilon = log_sig_min, log_sig_max, epsilon

        # Setup the shared layers
        if len(hidden_sizes):
            hidden_dim = hidden_sizes[-1]
            self.feat_x = mlp.MLP(in_features, hidden_sizes[-1], hidden_sizes[:-1], activation_class)
        else:
            hidden_dim = in_features
            self.feat_x = None

        # Setup the heads
        self.hidden_dim = hidden_dim
        self.mu_logstd = mlp.MLP(hidden_dim, out_features * 2, tuple(), activation_class)
        self.sde = nn.Linear(hidden_dim, out_features,bias=False)
        self.reset_sde()

    def reset_sde(self):
        # Ensure the sde conserves the variance of the input!
        nn.init.normal_(self.sde.weight,mean=0.,std=(1. / (self.hidden_dim ** 0.5)))

    def forward(self, state):
        if self.feat_x is not None:
            state = self.feat_x(state)

        # get gaussian parameters
        logits = self.mu_logstd(state)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max).exp()

        # sample the gaussian
        with torch.no_grad():
            state :Tensor = state - state.mean(dim=-1,keepdim=True) # zero mean
            state = state / state.std().clamp_min(1e-4) # unit variance
            sde_sample = self.sde(state)

        normal = distributions.Normal(mean, std)
        x_t = mean + (sde_sample * std) # reparametixation trick
        log_prob = normal.log_prob(x_t)

        # Enforce Action Bounds
        action = torch.tanh(x_t)
        log_prob = log_prob - torch.log((1 - action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


class DoubleQNetwork(nn.Module):
    # Creates 2 Q networks and concats their output over the last dimension
    def __init__(self, *args):
        super(DoubleQNetwork, self).__init__()
        self.nets = nn.ModuleList([mlp.MLP(*args), mlp.MLP(*args)])

    def forward(self, x):
        out = [net.forward(x) for net in self.nets]
        out = torch.cat(out, dim=-1)
        return out
