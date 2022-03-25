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
        self.visible_layer_encoders = nn.ModuleDict({})
        latent_dim = 0
        if "obs_2d" in obs_space.spaces:
            input_shape = obs_space.spaces["obs_2d"].shape
            # self.net["obs_2d"] = Atari(input_shape[0], conf.conv_channels)
            # self.net["obs_2d"] = AtariDataEfficient(input_shape[0], )
            # self.net["obs_2d"] = BoomNet(input_shape, conf.hidden_dim * 2, conf.conv_channels)
            example_input = torch.rand((1,) + tuple(input_shape), dtype=torch.float32)
            example_output: Tensor = self.visible_layer_encoders["obs_2d"](example_input)
            latent_dim += example_output.shape[-1]

        if any([x in obs_space.spaces for x in ["obs_1d", "achieved_goal"]]):
            input_shape = 0
            if "obs_1d" in obs_space.spaces:
                input_shape = np.prod(obs_space.spaces["obs_1d"].shape).item()
            if "desired_goal" in obs_space.spaces:
                input_shape += 2 * np.prod(obs_space.spaces["desired_goal"].shape).item()

            self.visible_layer_encoders["obs_1d"] = MLP(input_shape, conf.hidden_features, conf.obs_1d_hidden_dims)
            latent_dim += conf.hidden_features  # + input_shape

        self.conf = conf
        self.mode = conf.joiner_mode
        self.out_features = out_features
        if conf.joiner_mode == conf.JoinerModeEnum.feedforward:
            self.joiner = MLP(latent_dim, out_features, conf.joint_hidden_dims)
        elif conf.joiner_mode == conf.JoinerModeEnum.gru:
            self.joiner = nn.GRU(latent_dim, out_features, len(conf.joint_hidden_dims))
            self.hidden_state = torch.nn.Parameter(self.get_random_hidden(),requires_grad=True)
        else:
            raise ValueError(f"Unexpected value for {conf.joiner_mode}")


        self.param_dict:typing.Dict[str,typing.List[torch.Tensor]]= {}
        for m in self.visible_layer_encoders:
            self.param_dict[m] = list(self.visible_layer_encoders[m].parameters())
        self.param_dict["joiner"] = list(self.joiner.parameters())

    def forward(self, obs: TensorDict) -> Tensor:
        obs = dict(**obs)  # shallow copy dict
        if "achieved_goal" in obs:
            obs["obs_1d"] = torch.cat(
                ((obs["obs_1d"],) if "obs_1d" in obs else tuple()) +
                (obs["achieved_goal"], obs["desired_goal"]),
                dim=-1)
        encoder_outputs = [self.visible_layer_encoders[key](obs[key]) for key in
                           self.visible_layer_encoders]  # + ([obs["obs_1d"]] if "obs_1d" in obs else [])
        encoder_outputs = torch.cat(encoder_outputs, dim=-1)
        if self.mode == EncoderConf.JoinerModeEnum.feedforward:
            y = self.joiner(encoder_outputs)
            hidden = None
        elif self.mode == EncoderConf.JoinerModeEnum.gru:
            y, hidden = self.joiner(encoder_outputs, obs.get("agent_state", None))
        return y, hidden

    def reset(self):
        pass

    def forward_eval(self, x: TensorDict):
        for v in x.values(): v.unsqueeze_(0)
        y, hidden = self(x)
        for v in x.values(): v.squeeze_(0)
        return y.squeeze_(0), hidden.squeeze_(0) if hidden is not None else hidden

    def forward_train(self, x):
        conf = self.conf
        if self.mode == EncoderConf.joiner_mode.gru:
            x["is_contiguous"] = torch.cumprod(x["is_contiguous"], dim=0)
            scalar_shape = x["is_contiguous"].shape
            if conf.rnn_latent_state_training_mode == conf.RnnLatentStateTrainMode.store:
                x["agent_state"] = x["agent_state"][0].unsqueeze(0)
            elif conf.rnn_latent_state_training_mode == conf.RnnLatentStateTrainMode.learned:
                x["agent_state"] = self.hidden_state.view((1,1)+self.hidden_state.shape)
                x["agent_state"] = x["agent_state"].repeat(1,scalar_shape[1],1)
            elif conf.rnn_latent_state_training_mode == conf.RnnLatentStateTrainMode.zero:
                x["agent_state"] = None
            else:
                raise ValueError(f"Expected {conf.rnn_latent_state_training_mode} to be one of{conf.RnnLatentStateTrainMode}")
            y, h = self(x)
            del x["agent_state"]
            return y
        else:
            y, h = self(x)
            return y

    def get_random_hidden(self):
        if self.mode == EncoderConf.JoinerModeEnum.feedforward:
            return None
        elif self.mode == EncoderConf.joiner_mode.gru:
            return torch.rand((self.out_features))
