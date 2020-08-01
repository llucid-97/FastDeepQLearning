from .mlp import MLP
import torch
from torch import nn, Tensor, distributions


class GaussianMLP(MLP):
    # use an mlp to predict mean and std of gaussian, and use reparametization trick to sample it
    def __init__(self, in_features, out_features, hidden_sizes: tuple, activation_class=nn.ReLU,
                 log_sig_min=-20.0, log_sig_max=2, epsilon=1e-4):
        super(GaussianMLP, self).__init__(in_features, out_features * 2, hidden_sizes, activation_class)

        # define bounds on variance for numeric stability
        self.log_sig_min, self.log_sig_max, self.epsilon = log_sig_min, log_sig_max, epsilon

    def forward(self, state):
        # get gaussian parameters
        logits = super(GaussianMLP, self).forward(state)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

        # sample the gaussian
        normal = distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        log_prob = normal.log_prob(x_t)

        # Enforce Action Bounds
        action = torch.tanh(x_t)
        log_prob = log_prob - torch.log((1 - action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean
