from .models import make_actor, make_critic
import torch, numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from ...conf import AgentConf
from ...Utils import target_updates
import typing as T


class SoftActorCriticModule(nn.Module):
    def __init__(self, conf: AgentConf, input_dim):
        super().__init__()
        self.conf = conf

        # Actor Critic networks
        self.critic = make_critic(conf, input_dim)
        self.target_critic = make_critic(conf, input_dim)
        self.frozen_critic = make_critic(conf, input_dim)
        target_updates.hard_update(self.target_critic, self.critic)

        self.actor = make_actor(conf, input_dim)
        self.target_actor = make_actor(conf, input_dim)

        # Target Entropy parameter
        from gym.spaces import Discrete
        self.discrete = isinstance(conf.action_space, Discrete)
        if self.discrete:
            self.target_entropy = - self.conf.action_space.n
        else:
            self.target_entropy = - np.product(self.conf.action_space.shape).item()
        self.log_alpha = torch.nn.Parameter(torch.tensor(-2, dtype=torch.float32), requires_grad=True)
        self.curr_alpha = np.exp(self.log_alpha.item())

    def parameters(self, *args, **kwargs):
        # gives trainable parameters
        return (
                list(self.actor.parameters())
                + list(self.critic.parameters())
                + [self.log_alpha]
        )

    def update_target(self):
        update_fn = target_updates.soft_update if self.conf.use_soft_targets else target_updates.hard_update
        update_fn(self.target_actor, self.actor, self.conf.tau)
        update_fn(self.target_critic, self.critic, self.conf.tau)

    def act(self, state):
        return self.actor.forward(state)

    def critic_loss(self, curr_xp: T.Dict[str, Tensor], next_xp):
        """Expects tensors to be pre-sliced TD style!"""

        with torch.no_grad():
            # Get Action prediction from target policy network
            next_action, next_log_pi, _ = self.target_actor(next_xp["state"])
            entropy = -next_log_pi
            next_state_action = torch.cat((next_xp["state"], next_action), dim=-1)

            # Get TD target
            target_q = self.target_critic(next_state_action) + self.curr_alpha * entropy
            target_q, _ = torch.min(target_q, dim=-1, keepdim=True)  # for ensemble Q learning.
            td_target = next_xp["reward"] + next_xp["mask"] * self.conf.gamma * target_q
            assert td_target.shape == target_q.shape

        # Get online critic predictions
        if self.discrete:
            curr_xp["action"] = torch.eye(
                self.conf.action_space.n,
                device=curr_xp["action"].device, dtype=self.conf.dtype
            )[curr_xp["action"].view(curr_xp["action"].shape[:-1]).long()]
        state_action = torch.cat((curr_xp["state"], curr_xp["action"]), dim=-1)
        q_pred = self.critic(state_action)

        # Compute bellman loss. Ignore the warning: i'm just abusing the broadcast
        q_loss = F.smooth_l1_loss(q_pred, td_target, reduction="none")
        return q_loss

    def actor_loss(self, xp: T.Dict[str, Tensor]):
        pi, log_pi, _ = self.actor(xp["state"])

        # Note: we detach enc_state for the reason above but We still backprop into the encoder via pi.
        state_action = torch.cat((xp["state"].detach(), pi), dim=-1)

        self.frozen_critic.load_state_dict(self.critic.state_dict())
        qpi = self.frozen_critic(state_action)
        qpi, _ = torch.min(qpi, dim=-1, keepdim=True)

        entropy = -log_pi
        policy_loss = -(self.curr_alpha * entropy) - qpi
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach())

        self.curr_alpha = torch.exp(self.log_alpha).detach().requires_grad_(False)
        return policy_loss, alpha_loss, self.curr_alpha.detach()
