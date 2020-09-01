from .mlp import MLP
import torch
from torch import distributions, Tensor, nn
from torch.nn import functional as F


class GumbelMLP(MLP):
    # MLP that parametizes a gumbel softmax distribution
    def __init__(self, in_features, out_features, hidden_sizes: tuple, activation_class=nn.ReLU, temperature=1.0):
        super(GumbelMLP, self).__init__(in_features, out_features, hidden_sizes, activation_class)
        self.temperature = temperature

    def forward(self, x):
        logits: Tensor = super(GumbelMLP, self).forward(x)
        dist = GumbelSoftmax(temperature=torch.tensor(self.temperature, dtype=logits.dtype, device=logits.device),
                             logits=logits)
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        explore_action = x_t
        log_prob = dist.log_prob(x_t)

        return explore_action, log_prob, logits


class GumbelSoftmax(distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        '''
        Gumbel-softmax resampling using the Straight-Through trick.
        Credit to Ian Temple for bringing this to our attention. To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        '''
        rout = super().rsample(sample_shape)  # differentiable
        out = F.one_hot(torch.argmax(rout, dim=-1), self.logits.shape[-1]).float()
        return (out - rout).detach() + rout

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1, keepdim=True)
