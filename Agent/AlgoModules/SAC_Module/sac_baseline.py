"""
This is a modified version of Pranjal Tandon's Pytorch Soft Actor Critic [https://github.com/pranz24/pytorch-soft-actor-critic]

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
from .models import make_actor, make_critic
import torch, numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from ...conf import AgentConf
from ...Utils import target_updates
import typing as T


class SoftActorCriticModule(nn.Module):
    def __init__(self, conf: AgentConf, input_dim, critic_factory=make_critic, actor_factory=make_actor):
        super().__init__()
        self.conf = conf

        # Actor Critic networks
        self.critic = critic_factory(conf, input_dim)
        self.target_critic = critic_factory(conf, input_dim)
        self.frozen_critic = critic_factory(conf, input_dim)
        target_updates.hard_update(self.target_critic, self.critic)

        self.actor = actor_factory(conf, input_dim)
        self.target_actor = actor_factory(conf, input_dim)

        # Target Entropy parameter
        from gym.spaces import Discrete
        self.discrete = isinstance(conf.action_space, Discrete)
        if self.discrete:
            self.target_entropy = - self.conf.action_space.n
        else:
            self.target_entropy = - np.product(self.conf.action_space.shape).item()
        self.log_alpha = torch.nn.Parameter(torch.tensor(conf.initial_log_alpha, dtype=torch.float32),
                                            requires_grad=True)
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
        summaries = {}
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
            # Make it a OneHot Vector to match format of NN outputs
            action = torch.eye(self.conf.action_space.n,
                               device=self.conf.training_device, dtype=self.conf.dtype
                               )[curr_xp["action"].view(curr_xp["action"].shape[:-1]).long()]
        else:
            action = curr_xp["action"]
        state_action = torch.cat((curr_xp["state"], action), dim=-1)
        q_pred = self.critic(state_action)

        # Compute bellman loss. Ignore the warning: i'm just abusing the broadcast
        q_loss = F.smooth_l1_loss(q_pred, td_target, reduction="none")
        # q_loss = F.mse_loss(q_pred, td_target, reduction="none")

        bootstrap_n_step_lowerbound = None
        if self.conf.use_nStep_lowerbounds:
            # Use the sampled return as a lower bound for Q predictions
            lowerbound = (next_xp["n_step_return"] - q_pred).relu_()
            lb_mask = (lowerbound == 0)
            q_loss = (q_loss * lb_mask) + lowerbound
            with torch.no_grad():
                summaries["mc_constraint_violations"] = torch.logical_not(lb_mask).sum().float() / np.prod(
                    lb_mask.shape).item()

            if self.conf.use_bootstrap_nstep:
                # Bootstrapped n-step return
                # Uses the target network to predict remainder of score beyond the temporal window
                td_len = self.conf.temporal_len - 1
                # Temporal Consistency Penalties
                # By definition of Q*, Q*(s0,a0) must be greater than Q*(sn,an)
                # We can exploit this property to construct bounds on Q predictions over >1 time step
                with torch.no_grad():
                    # Calculate the n-step reward up to our time horizon
                    try:
                        window_discounts = self._memoized_window_discounts  # memoize it so we don't have to recompute every cycle
                    except AttributeError:
                        window_discounts = self.conf.gamma ** torch.arange(td_len, device=self.conf.training_device,
                                                                           dtype=self.conf.dtype).view(-1, 1, 1)
                        self._memoized_window_discounts = window_discounts
                    window_R = (next_xp["reward"] * window_discounts).sum(0)  # n-step return over window
                    window_mask = next_xp["mask"].prod(0)

                # penalize any points where the bound > pred. Only done for first element
                bootstrap_n_step_lowerbound = window_mask * (
                        (window_R + ((self.conf.gamma ** td_len) * td_target[-1]))
                        - q_pred[0]
                ).relu_()
                with torch.no_grad():
                    summaries["n_step_bootstrap_violations"] = torch.logical_not(
                        bootstrap_n_step_lowerbound == 0).sum().float() / np.prod(bootstrap_n_step_lowerbound.shape).item()

        return q_loss, bootstrap_n_step_lowerbound, summaries

    def actor_loss(self, xp: T.Dict[str, Tensor]):
        summaries = {}
        pi, log_pi, _ = self.actor(xp["state"])

        # Note: we detach enc_state for the reason above but We still backprop into the encoder via pi.
        state_action = torch.cat((xp["state"].detach(), pi), dim=-1)

        self.frozen_critic.load_state_dict(self.critic.state_dict())
        qpi = self.frozen_critic(state_action)
        qpi, _ = torch.min(qpi, dim=-1, keepdim=True)

        entropy = -log_pi
        policy_loss = -(self.curr_alpha * entropy) - qpi
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach())

        self.curr_alpha = torch.exp(self.log_alpha).detach()
        summaries["curr_alpha"] = self.curr_alpha
        return policy_loss, alpha_loss, summaries
