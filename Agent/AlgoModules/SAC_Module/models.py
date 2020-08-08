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


class DoubleQNetwork(nn.Module):
    # Creates 2 Q networks and concats their output over the last dimension
    def __init__(self, *args):
        super(DoubleQNetwork, self).__init__()
        self.nets = nn.ModuleList([mlp.MLP(*args), mlp.MLP(*args)])

    def forward(self, x):
        out = [net.forward(x) for net in self.nets]
        out = torch.cat(out, dim=-1)
        return out
