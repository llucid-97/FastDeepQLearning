"""
MIT License

Copyright (c) 2018 Pranjal Tandon
Copyright (c) 2020 Gershom Agim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from .soft_actor_critic import SoftActorCritic
import torch
from franQ.Agent.models import mlp
# from franQ.Agent.models import gumbel_mlp
from franQ.Agent.conf import AgentConf
from torch import nn, Tensor, distributions


def make_sde_actor(conf: AgentConf, input_dim):
    from gym.spaces import Discrete
    if isinstance(conf.action_space, Discrete):
        raise NotImplementedError(
            "TODO: Add this once gaussian is in working state and gumbel softmax policy has been tested in regular SAC!")
        # return gumbel_softmax.GumbelMLP(input_dim, conf.action_space.n, conf.mlp_hidden_dims, conf.mlp_activation)
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
        self.sde = nn.Linear(hidden_dim, out_features, bias=False)
        self.reset_sde()

    def reset_sde(self):
        # Ensure the sde conserves the variance of the input!
        nn.init.normal_(self.sde.weight, mean=0., std=(1. / (self.hidden_dim ** 0.5)))

    def forward(self, state):
        if self.feat_x is not None:
            state = self.feat_x(state)

        # get gaussian parameters
        logits = self.mu_logstd(state)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max).exp()

        # sample the gaussian
        normal = distributions.Normal(mean, std)

        if self.training:
            x_t = normal.rsample()
        else:
            with torch.no_grad():
                state: Tensor = state - state.mean(dim=-1, keepdim=True)  # zero mean
                state = state / state.std().clamp_min(1e-4)  # unit variance
                sde_sample = self.sde(state)

            x_t = mean + (sde_sample * std)  # reparametixation trick
        log_prob = normal.log_prob(x_t)

        # Enforce Action Bounds
        action = torch.tanh(x_t)
        log_prob = log_prob - torch.log((1 - action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


class SDESoftActorCriticModule(SoftActorCritic):
    """Soft actor critic with state dependent exploration"""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SDE is not yet in a working state. DO not use. Only merged changes to stash progress")
        # super(SDESoftActorCriticModule, self).__init__(*args, **kwargs, actor_factory=make_sde_actor)
        # self._step = 0

    def update_target(self):
        super(SDESoftActorCriticModule, self).update_target()
        self.actor.reset_sde()
        self.target_actor.reset_sde()

    def act(self, state):
        if (self._step % self.conf.sde_update_interval) == 0: self.actor.reset_sde()
        return self.actor.forward(state)
