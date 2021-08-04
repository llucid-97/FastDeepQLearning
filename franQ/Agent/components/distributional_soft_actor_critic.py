"""
This is based on SamsungLabs implementation of TQC: https://github.com/SamsungLabs/tqc_pytorch
used under MIT License:
MIT License

Copyright (c) 2020 Samsung

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
import torch
from torch import Tensor
import numpy as np
import typing as T

from .soft_actor_critic import SoftActorCritic


class DistributionalSoftActorCritic(SoftActorCritic):
    """
    Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics.
    Paper: https://arxiv.org/abs/2005.0426
    """

    def q_loss(self, curr_xp: T.Dict[str, Tensor], next_xp: T.Dict[str, Tensor]):
        conf = self.conf
        summaries = {}
        with torch.no_grad():
            # Get Deep Q Learning targets
            (next_action, next_log_pi, _) = self.actor_target(next_xp["state"])
            entropy = -next_log_pi

            next_state_action = torch.cat((next_xp["state"], next_action), dim=-1)
            next_z = self.critic_target(next_state_action)
            sorted_z, _ = torch.sort(next_z, dim=-1)
            target_q = sorted_z[
                       ...,
                       :-int(conf.top_quantiles_to_drop * sorted_z.shape[-1])]
            if conf.use_max_entropy_q:
                target_q = target_q + self.curr_alpha * entropy

            # Handle over-estimation bias
            td_target = next_xp["reward"] + next_xp["mask"] * conf.gamma * target_q
            assert td_target.shape == target_q.shape
            # TODO: Switch to using callback functions for logging and add histograms for current q and target

        # Get online critic predictions
        action = curr_xp["action_onehot"] if self.conf.discrete else curr_xp["action"]
        q_pred: Tensor = self.critic(torch.cat((curr_xp["state"], action), dim=-1))
        with torch.no_grad():
            summaries["q_pred_mu"] = q_pred.mean()
            summaries["q_pred_var"] = q_pred.var(-1).mean()

        # Compute bellman loss. Ignore the warning: i'm just abusing the broadcast
        q_loss = quantile_huber_loss_f(q_pred, td_target).unsqueeze(-1)
        # q_loss = F.mse_loss(q_pred, td_target, reduction="none")

        # TODO: Move this to a different function to avoid code duplication. Ensure source SAC has same q_loss dims as this; I've changed those
        # Get aux losses & Optimality tightening
        bootstrap_minibatch_nstep_lowerbound = None
        if self.conf.use_nStep_lowerbounds:
            # Use the sampled return as a lower bound for Q predictions
            lowerbound = (next_xp["mc_return"] - q_pred).relu_()
            q_loss = q_loss + lowerbound.mean(-1, keepdim=True)
            with torch.no_grad():
                summaries["mc_constraint_violations"] = (lowerbound > 0).sum().float() / np.prod(
                    lowerbound.shape).item()

            if conf.use_bootstrap_minibatch_nstep:
                raise NotImplementedError("Need to update this from SAC to use the quantile huber loss!")

        return q_loss, bootstrap_minibatch_nstep_lowerbound, summaries


def quantile_huber_loss_f(quantiles: torch.Tensor, samples, ):
    pairwise_delta = samples[..., None, :] - quantiles[..., None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[-1]
    tau = torch.arange(n_quantiles, device=quantiles.device, dtype=quantiles.dtype) / n_quantiles + 1 / 2 / n_quantiles
    shape = [1 for _ in quantiles.shape] + [1]
    shape[-2] = n_quantiles
    tau = torch.reshape(tau, shape)
    loss = (torch.abs(tau - (pairwise_delta < 0).float()) * huber_loss).mean((-1, -2))
    return loss
