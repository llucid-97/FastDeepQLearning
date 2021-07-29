import torch, typing
from torch import nn, Tensor
from franQ.Agent.models.mlp import MLP
from franQ.Agent.conf import AgentConf
import numpy as np


class Encoder(nn.Module):
    def __init__(self, conf: AgentConf):
        super().__init__()
        self.net = nn.ModuleDict({})
        latent_dim = 0
        if "obs_2d" in conf.obs_space.spaces:
            input_shape = conf.obs_space.spaces["obs_2d"].shape
            # self.net["obs_2d"] = Atari(input_shape[0], conf.conv_channels)
            # self.net["obs_2d"] = AtariDataEfficient(input_shape[0], )
            # self.net["obs_2d"] = BoomNet(input_shape, conf.hidden_dim * 2, conf.conv_channels)
            example_input = torch.rand((1,) + tuple(input_shape), dtype=torch.float32)
            example_output: Tensor = self.net["obs_2d"](example_input)
            latent_dim += example_output.shape[-1]

        input_shape = 0
        if any([x in conf.obs_space.spaces for x in ["obs_1d", "achieved_goal"]]):
            try:
                input_shape = np.prod(conf.obs_space.spaces["obs_1d"].shape).item()
            except KeyError:
                pass
            try:
                input_shape += 2 * np.prod(conf.obs_space.spaces["desired_goal"].shape).item()
            except KeyError:
                pass
            from franQ.Agent.models.identity import Identity

            self.net["obs_1d"] = MLP(input_shape, conf.latent_state_dim, conf.enc1d_hidden_dims)
            latent_dim += conf.latent_state_dim #+ input_shape
            # self.net["obs_1d"] = Identity()
            # latent_dim += input_shape


        self.joiner = MLP(latent_dim, conf.latent_state_dim, [])

    def forward(self, obs: typing.Dict[str, Tensor]) -> Tensor:
        obs = dict(**obs)  # shallow copy dict
        if "achieved_goal" in obs:
            obs["obs_1d"] = torch.cat(
                ((obs["obs_1d"],) if "obs_1d" in obs else tuple()) +
                (obs["achieved_goal"], obs["desired_goal"]),
                dim=-1)
        encoder_outputs = [self.net[key](obs[key]) for key in self.net] #+ ([obs["obs_1d"]] if "obs_1d" in obs else [])
        encoder_outputs = torch.cat(encoder_outputs, dim=-1)
        latent_state = self.joiner(encoder_outputs)
        # latent_state = scale_norm(latent_state)
        return latent_state

    def reset(self):
        pass

    def forward_eval(self, x):
        return self.forward(x)

    def forward_train(self, x, **kwargs):
        return self.forward(x)
