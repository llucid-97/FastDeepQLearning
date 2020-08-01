import torch, numpy as np
from torch import Tensor, nn, distributions
import typing as T
from Agent.conf import AgentConf


class RandomNetworkAgent(nn.Module):
    def __init__(self, conf: AgentConf,replays):
        super(RandomNetworkAgent, self).__init__()
        self.conf = conf
        try:
            num_actions = conf.action_space.shape[-1]
            self.discrete= False
        except AttributeError:
            num_actions = conf.action_space.n
            self.discrete = True
        self.net = nn.Linear(conf.obs_space.shape[-1], num_actions)

    @property
    def iteration(self):
        return 0


    def act(self, exp: T.Dict[str, Tensor]) -> T.Dict[str, np.ndarray]:
        with torch.no_grad():
            action :torch.Tensor = self.net(exp["obs"])
            if self.discrete:
                action = action.multinomial(1)
            else:
                action = action.tanh_()
            return action