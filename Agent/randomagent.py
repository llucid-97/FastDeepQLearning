from torch import nn, multiprocessing as mp, Tensor
import typing as T
from Agent.conf import AgentConf
from collections import OrderedDict
import torch
import logging


class RandomAgent(nn.Module):
    def __init__(self, conf: AgentConf, replays, **kwargs):
        nn.Module.__init__(self)
        self.conf = conf
        param_queue: T.Union[mp.Queue, T.List[mp.Queue]]
        self.param_queue = kwargs.get("param_queue",mp.Queue(maxsize=1))

        # Logging
        from torch.utils.tensorboard import SummaryWriter
        from pathlib import Path

        self.summary_writer = SummaryWriter(
            Path(conf.log_dir) / "rl_algo"
        )

        try:
            num_actions = conf.action_space.shape[-1]
            self.discrete= False
        except AttributeError:
            num_actions = conf.action_space.n
            self.discrete = True

        latent_state_dim = 0
        encoders = {}
        if "obs_1d" in conf.obs_space.spaces:
            encoders["obs_1d"] = nn.Linear(conf.obs_space.spaces["obs_1d"].shape[-1], conf.latent_state_dim)
            latent_state_dim += conf.latent_state_dim

        if "obs_2d" in conf.obs_space.spaces:
            encoders["obs_2d"] = nn.Linear(1,1)
            latent_state_dim +=1
        self.encoders = nn.ModuleDict(encoders)
        self.net = nn.Linear(latent_state_dim, num_actions)
        self.step = 0

    @property
    def iteration(self):
        self.step +=1
        return self.step
    def act(self, exp: T.Dict[str, Tensor]):
        with torch.no_grad():
            latent = []
            if "obs_1d" in exp:
                latent.append(self.encoders["obs_1d"](exp["obs_1d"]))
            if "obs_2d" in exp:
                img = exp["obs_2d"].view(exp["obs_2d"].shape[0],-1).mean(-1)
                latent.append(self.encoders["obs_2d"](img))

            latent = torch.cat(latent,dim=-1)
            action: torch.Tensor = self.net(latent)
            if self.discrete:
                action = action.multinomial(1)
            else:
                action = action.tanh_()
            return action

    def pull_params(self):
        if not self.param_queue.empty():
            params = self.param_queue.get()
            self.load_state_dict(params)

    @staticmethod
    def dict_to(state_dict: OrderedDict, device=None):
        device = torch.device("cpu:0") if device is None else device
        return OrderedDict({k: v.to(device) for k, v in state_dict.items()})

    def push_params(self):
        if not self.param_queue.full():
            self.param_queue.put(self.dict_to(self.state_dict(), "cpu:0"))



    def update_targets(self):
        pass

    def reset(self):
        pass

    def get_losses(self, experience_dict):
        sequence_len = experience_dict["episode_step"].shape[0]
        temporal_continuity = experience_dict["episode_step"][1:] == (experience_dict["episode_step"][:-1] + 1)
        temporal_continuity &= experience_dict["episode_done"][1:]

        zero = torch.zeros(1, device=temporal_continuity.device, requires_grad=True)
        zero = zero * 2
        return zero
