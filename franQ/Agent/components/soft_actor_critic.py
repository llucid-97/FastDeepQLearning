import torch
from torch import nn, Tensor, jit, distributions
from torch.nn import functional as F
import numpy as np
from franQ.Agent.models import mlp, gumbel_mlp
from franQ.Agent.models import gaussian_mlp
from franQ.Agent.utils.common import hard_update, soft_update
from franQ.Agent.conf import AgentConf
import typing as T


def make_actor(conf: AgentConf, input_dim):
    if conf.discrete:
        pi = gumbel_mlp.GumbelMLP(input_dim, conf.action_space.n, conf.mlp_hidden_dims)
    else:
        pi = gaussian_mlp.GaussianMLP(input_dim, conf.action_space.shape[0], conf.mlp_hidden_dims)
    return pi


def make_critic(conf: AgentConf, input_dim):
    input_dim = input_dim + (conf.action_space.n if conf.discrete else conf.action_space.shape[-1])
    return mlp.MLPEnsemble(input_dim, conf.num_q_predictions, conf.mlp_hidden_dims,
                           ensemble_size=conf.num_q_networks)


class SoftActorCritic(nn.Module):
    def __init__(self, conf: AgentConf, input_dim):
        nn.Module.__init__(self)
        self.conf = conf

        self.critic = make_critic(conf, input_dim)
        self.critic_target = make_critic(conf, input_dim)
        self.critic_frozen = make_critic(conf, input_dim)  # Note: This is used for updating the actor
        hard_update(self.critic_target, self.critic)

        self.actor = make_actor(conf, input_dim)
        self.actor_target = make_actor(conf, input_dim)
        hard_update(self.actor_target, self.actor)

        self.log_alpha = torch.nn.Parameter(torch.tensor(conf.init_log_alpha, dtype=torch.float32), requires_grad=True)
        self.curr_alpha = np.exp(self.log_alpha.item())
        self.target_entropy = -(conf.action_space.n if conf.discrete else np.product(conf.action_space.shape).item())

        self._fast_params = (list(self.actor.parameters()) +
                             list(self.critic.parameters()) +
                             [self.log_alpha])

    def parameters(self, *args, **kwargs):  # get trainable parameters
        return self._fast_params

    def update_target(self):
        if not self.conf.use_hard_updates:
            soft_update(self.actor_target, self.actor, self.conf.tau)
            soft_update(self.critic_target, self.critic, self.conf.tau)
        else:
            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_target, self.critic)

    def act(self, state):
        explore_action, log_prob, greedy_action = self.actor(state)
        if self.conf.discrete:
            # Outputs are distribution (or logits) over actions. Must sample
            explore_action: Tensor = explore_action.argmax(-1, True)  # argmax is fine: already sampled gumbel softmax
            greedy_action: Tensor = greedy_action.argmax(-1, True)
        return explore_action, log_prob, greedy_action

    def q_loss(self, curr_xp: T.Dict[str, Tensor], next_xp: T.Dict[str, Tensor]):
        conf = self.conf
        summaries = {}
        with torch.no_grad():
            # Get Deep Q Learning targets
            (next_action, next_log_pi, _) = self.actor_target(next_xp["state"])
            entropy = -next_log_pi

            next_state_action = torch.cat((next_xp["state"], next_action), dim=-1)
            target_q = self.critic_target(next_state_action)
            if conf.use_max_entropy_q:
                target_q = target_q + self.curr_alpha * entropy

            # Handle over-estimation bias
            target_q, _ = torch.min(target_q, dim=-1, keepdim=True)  # for ensemble Q learning & multi-pred
            td_target = next_xp["reward"] + next_xp["mask"] * conf.gamma * target_q
            assert td_target.shape == target_q.shape

        # Get online critic predictions
        action = curr_xp["action_onehot"] if self.conf.discrete else curr_xp["action"]
        q_pred: Tensor = self.critic(torch.cat((curr_xp["state"], action), dim=-1))
        summaries["q_pred_mu"] = q_pred.mean()
        summaries["q_pred_var"] = q_pred.var()

        # Compute bellman loss. Ignore the warning: i'm just abusing the broadcast
        q_loss = F.smooth_l1_loss(q_pred, td_target, reduction="none")
        # q_loss = F.mse_loss(q_pred, td_target, reduction="none")

        # Get aux losses & Optimality tightening
        bootstrap_minibatch_nstep_lowerbound = None
        if self.conf.use_nStep_lowerbounds:
            # Use the sampled return as a lower bound for Q predictions
            lowerbound = (next_xp["mc_return"] - q_pred).relu_()
            lb_mask = (lowerbound == 0)
            q_loss = (q_loss * lb_mask) + lowerbound
            with torch.no_grad():
                summaries["mc_constraint_violations"] = torch.logical_not(lb_mask).sum().float() / np.prod(
                    lb_mask.shape).item()

            if conf.use_bootstrap_minibatch_nstep:
                # Calculate the n-step return over the whole temporal window.
                # then bootstrap from the target network to approximate the rest
                # Use this as a lowerbound for the Q prediction at t=0 in minibatch.
                with torch.no_grad():
                    try:  # memoize it so we don't have to recompute every cycle
                        gamma_arange = self._gamma_arange
                    except AttributeError:
                        gamma_arange = conf.gamma ** torch.arange(
                            conf.temporal_len - 1,
                            device=conf.training_device,
                            dtype=conf.dtype
                        ).view(-1, *[1 for _ in next_xp["reward"].shape[1:]])
                        self._gamma_arange = gamma_arange

                    minibatch_return = (next_xp["reward"] * gamma_arange).sum(0)
                    minibatch_mask = next_xp["mask"].prod(0)  # validity check: minibatch sequence must be contiguous

                # penalize any points where the bound > pred. Only done for first element
                bootstrap_minibatch_nstep_lowerbound = (
                        minibatch_mask  # Ensure we're only doing this for sequences in the same episode
                        * ((minibatch_return + ((conf.gamma ** (conf.temporal_len - 1)) * td_target[-1]))  # Bellman
                           - q_pred[0]  # prediction
                           ).relu_()  # Penalize any where the prediction is less than the lowerbound
                )

                # Tensorboard logs
                with torch.no_grad():
                    summaries["bootstrap_minibatch_nstep_violations"] = torch.logical_not(
                        bootstrap_minibatch_nstep_lowerbound == 0).sum().float() / np.prod(
                        bootstrap_minibatch_nstep_lowerbound.shape).item()

        return q_loss.mean(-1, keepdim=True), bootstrap_minibatch_nstep_lowerbound, summaries

    def actor_loss(self, xp: T.Dict[str, Tensor]):
        summaries = {}

        pi, log_pi, _ = self.actor(xp["state"])
        entropy = -log_pi

        self.critic_frozen.load_state_dict(self.critic.state_dict())
        state_action = torch.cat((xp["state"].detach(), pi), dim=-1)

        qpi = self.critic_frozen(state_action)
        qpi, _ = torch.min(qpi, dim=-1, keepdim=True)

        policy_loss = -(self.curr_alpha * entropy) - qpi
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach())
        self.curr_alpha = torch.exp(self.log_alpha).detach()
        summaries["curr_alpha"] = self.curr_alpha
        return policy_loss.mean(-1, keepdim=True), alpha_loss, summaries
