import torch, typing
from torch import nn, Tensor
from franQ.Agent.models.mlp import MLP
from franQ.Agent.conf import AgentConf, EncoderConf
import numpy as np
from franQ.Agent.models.identity import Identity

TensorDict = typing.Dict[str, Tensor]


class Encoder(nn.Module):
    def __init__(self, obs_space, out_features, conf: EncoderConf):
        super().__init__()
        self.net = nn.ModuleDict({})
        latent_dim = 0
        if "obs_2d" in obs_space.spaces:
            input_shape = obs_space.spaces["obs_2d"].shape
            # self.net["obs_2d"] = Atari(input_shape[0], conf.conv_channels)
            # self.net["obs_2d"] = AtariDataEfficient(input_shape[0], )
            # self.net["obs_2d"] = BoomNet(input_shape, conf.hidden_dim * 2, conf.conv_channels)
            example_input = torch.rand((1,) + tuple(input_shape), dtype=torch.float32)
            example_output: Tensor = self.net["obs_2d"](example_input)
            latent_dim += example_output.shape[-1]

        if any([x in obs_space.spaces for x in ["obs_1d", "achieved_goal"]]):
            input_shape = 0
            if "obs_1d" in obs_space.spaces:
                input_shape = np.prod(obs_space.spaces["obs_1d"].shape).item()
            if "desired_goal" in obs_space.spaces:
                input_shape += 2 * np.prod(obs_space.spaces["desired_goal"].shape).item()

            self.net["obs_1d"] = MLP(input_shape, conf.hidden_features, conf.obs_1d_hidden_dims)
            latent_dim += conf.hidden_features  # + input_shape

        if conf.mode == conf.ModeEnum.feedforward:
            self.joiner = MLP(latent_dim, out_features, conf.joint_hidden_dims)
        elif conf.mode == conf.ModeEnum.rnn:
            self.joiner = nn.RNN(latent_dim, out_features, len(conf.joint_hidden_dims),
                                 nonlinearity='relu')
        else:
            raise ValueError(f"Unexpected value for {conf.mode}")
        self.mode = conf.mode
        self.out_features = out_features

    def forward(self, obs: TensorDict) -> Tensor:
        obs = dict(**obs)  # shallow copy dict
        if "achieved_goal" in obs:
            obs["obs_1d"] = torch.cat(
                ((obs["obs_1d"],) if "obs_1d" in obs else tuple()) +
                (obs["achieved_goal"], obs["desired_goal"]),
                dim=-1)
        encoder_outputs = [self.net[key](obs[key]) for key in
                           self.net]  # + ([obs["obs_1d"]] if "obs_1d" in obs else [])
        encoder_outputs = torch.cat(encoder_outputs, dim=-1)
        if self.mode == EncoderConf.ModeEnum.feedforward:
            y = self.joiner(encoder_outputs)
            hidden = None
        elif self.mode == EncoderConf.ModeEnum.rnn:
            y, hidden = self.joiner(encoder_outputs, obs["agent_state"])
        return y, hidden

    def reset(self):
        pass

    def forward_eval(self, x: TensorDict):
        for v in x.values(): v.unsqueeze_(0)
        y, hidden = self(x)
        for v in x.values(): v.squeeze_(0)
        return y.squeeze_(0), hidden.squeeze_(0)

    def forward_train(self, x):
        if self.mode == EncoderConf.mode.rnn:
            x["is_contiguous"] = torch.cumprod(x["is_contiguous"], dim=0)
        x["agent_state"] = None
        y,h = self(x)
        del x["agent_state"]
        return y

    def get_random_hidden(self):
        if self.mode == EncoderConf.ModeEnum.feedforward:
            return None
        elif self.mode == EncoderConf.ModeEnum.rnn:
            return torch.rand((self.out_features))
