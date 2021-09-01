from torch import nn, multiprocessing as mp, Tensor

from franQ.Replay.wrappers import TorchDataLoader

import itertools

import typing as T, logging
from franQ.Agent.conf import AgentConf, AttrDict
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from threading import Thread
from queue import Queue

from .utils.common import soft_update, hard_update
from .components import soft_actor_critic, encoder

TensorDict = T.Dict[str, Tensor]


class DeepQLearning(nn.Module):
    def __init__(self, conf: AgentConf, replays, **kwargs):
        nn.Module.__init__(self)
        conf: AgentConf = conf if isinstance(conf, AttrDict) else AttrDict().from_dict(conf)
        self.conf = conf
        self.param_queue = kwargs.get("param_queue", mp.Queue(maxsize=1))

        if conf.use_async_train:
            if not kwargs.get("train_process", False):
                # make another process for doing parameter updates asynchronously
                mp.Process(target=DeepQLearning, args=[conf.to_dict(), replays],
                           kwargs={"param_queue": self.param_queue,
                                   "train_process": True}).start()
                Thread(target=self._pull_params).start()  # grab updated params from the other process
        else:
            # Sync mode means rollout and trainer are in same process, so must be on same device
            conf.training_device = conf.inference_device

        # Logging
        self.summary_writer = SummaryWriter(Path(conf.log_dir) / ("Trainer" if "train_process" in kwargs else "Actor"))
        self.fast_params = []
        # Models

        # Default encoder type
        self.encoder = encoder.Encoder(conf)
        self.target_encoder = encoder.Encoder(conf) if conf.use_target_encoder else self.encoder
        hard_update(self.target_encoder, self.encoder)
        self.fast_params += list(self.encoder.parameters())

        if conf.use_distributional_sac:
            from .components.distributional_soft_actor_critic import DistributionalSoftActorCritic
            self.actor_critic = DistributionalSoftActorCritic(conf, conf.latent_state_dim)
        else:
            self.actor_critic = soft_actor_critic.SoftActorCritic(conf, conf.latent_state_dim)
        self.fast_params += list(self.actor_critic.parameters())

        self.step = 0
        train_proc = kwargs.get("train_process", False)
        if train_proc or (not conf.use_async_train):
            if not train_proc:
                conf.inference_device = conf.training_device
            self.replays = [TorchDataLoader(r, conf.training_device, conf.dtype) for r in replays]
            self.to(conf.training_device)
            self.optimizers = [torch.optim.Adam(
                self.parameters(),
                lr=self.conf.learning_rate
            )]

            if kwargs.get("train_process", False):
                dump_q = Queue(maxsize=1)
                Thread(target=self._push_params, args=[dump_q]).start()

                for step_train in itertools.count():
                    self.train_step()
                    if (step_train % self.conf.param_update_interval) == 0:
                        # Signal the _push_params thread that we've updated enough times to warrant a push
                        if dump_q.empty(): dump_q.put_nowait(None)

    def train_step(self):
        for replay in self.replays:
            experience_dict = replay.temporal_sample()
            task_loss = self.get_losses(experience_dict)

            [o.zero_grad() for o in self.optimizers]
            task_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.conf.clip_grad_norm)
            [o.step() for o in self.optimizers]
            self.update_targets()
            self.reset()
            self.conf.global_step.value += 1

    @property
    def iteration(self):
        return int(self.conf.global_step.value)

    def parameters(self, *args, **kwargs):
        return self.fast_params

    def _push_params(self, q: Queue):
        while True:
            _ = q.get()
            state_dict = self.state_dict()
            state_dict = OrderedDict({k: v.to("cpu:0") for k, v in state_dict.items()})
            self.param_queue.put(state_dict)

    def _pull_params(self):
        while True:
            params = self.param_queue.get()
            self.load_state_dict(params), logging.info("loaded state dict")

    @staticmethod
    def dict_to(state_dict: OrderedDict, device=None):
        device = torch.device("cpu:0") or device
        return OrderedDict({k: v.to(device) for k, v in state_dict.items()})

    def act(self, experiences: T.Dict[str, Tensor]):
        """Run agent inference."""
        if not self.conf.use_async_train:
            #  If async mode is disabled, take a training step here!
            if all([r.ready() for r in self.replays]):
                self.train_step()

        with torch.no_grad():
            latent_state = self.encoder.forward_eval(experiences)
            explore_action, log_prob, exploit_action = self.actor_critic.act(latent_state)

            if self.conf.num_instances > 1:
                # Vectorized Choose: whether to explore or exploit
                exploit_mask = (experiences["idx"] == 0).view(-1, 1)
                action = (exploit_action * exploit_mask) + (explore_action * torch.logical_not(exploit_mask))
            else:
                action = explore_action  # explore always if we only have 1 environment

            # if self.conf.discrete: action = action.argmax(-1, True)  # go from one-hot encoding to sparse
            return action

    def reset(self):
        self.encoder.reset()

    def update_targets(self):
        if self.conf.use_hard_updates:
            # hard updates should only be done once every N steps.
            if self.step % self.conf.hard_update_interval: return
            update = hard_update
        else:
            update = soft_update
        self.actor_critic.update_target()
        if self.conf.use_target_encoder:
            update(self.target_encoder, self.encoder, self.conf.tau)

    def _get_encodings_training(self, experience: TensorDict):
        enc_kwargs = {
            # "is_contiguous": is_contiguous,
            "sequence_len": self.conf.temporal_len,
            "batch_size": self.conf.batch_size
        }
        if self.conf.use_target_encoder:
            # Split then run both
            curr_xp, next_xp = self._temporal_difference_shift(experience)
            curr_xp["state"] = self.encoder.forward_train(curr_xp, **enc_kwargs)
            with torch.no_grad():
                next_xp["state"] = self.target_encoder.forward_train(next_xp, **enc_kwargs)
        else:
            # Run on all then split
            experience["state"] = self.encoder.forward_train(experience, **enc_kwargs)
            curr_xp, next_xp = self._temporal_difference_shift(experience)
        return curr_xp, next_xp

    def get_losses(self, experience: TensorDict):
        experience["mask"] = torch.logical_not(experience["task_done"])
        is_contiguous = experience["episode_step"][1:] == (experience["episode_step"][:-1] + 1)
        with torch.no_grad():
            # Convert discrete actions to 1-hot encoding
            if self.conf.discrete:
                experience["action_onehot"] = torch.eye(
                    self.conf.action_space.n,
                    device=experience["action"].device, dtype=experience["action"].dtype
                )[experience["action"].view(experience["action"].shape[:-1]).long()]

        curr_xp, next_xp = self._get_encodings_training(experience)

        q_loss, bootstrapped_lowerbound_loss, q_summaries = self.actor_critic.q_loss(curr_xp, next_xp)
        pi_loss, alpha_loss, pi_summaries = self.actor_critic.actor_loss(curr_xp)


        loss = ((q_loss + pi_loss + alpha_loss) * is_contiguous).mean()  # Once its recurrent, they all use TD
        if self.conf.use_bootstrap_minibatch_nstep:
            bootstrapped_lowerbound_loss = (bootstrapped_lowerbound_loss * is_contiguous.prod(0)).mean()
            loss = loss + bootstrapped_lowerbound_loss

        # Step 11: Write Scalars
        if (self.step % self.conf.log_interval) == 0:
            assert q_loss.shape == pi_loss.shape == is_contiguous.shape == alpha_loss.shape, \
                f"loss shape mismatch: q={q_loss.shape} pi={pi_loss.shape} c={is_contiguous.shape} a={alpha_loss.shape}"
            self.summary_writer.add_scalars("Trainer/RL_Loss", {"Critic": q_loss.mean().item(),
                                                                "Actor": pi_loss.mean().item(),
                                                                "Alpha": alpha_loss.mean().item(), },
                                            self.step)

            [self.summary_writer.add_scalar(f"Trainer/Critic_{k}", v, self.step) for k, v in q_summaries.items()]
            [self.summary_writer.add_scalar(f"Trainer/Actor_{k}", v, self.step) for k, v in pi_summaries.items()]

        self.step += 1

        return loss / self.conf.temporal_len

    @staticmethod
    def _temporal_difference_shift(experience_dict: TensorDict) -> T.Tuple[TensorDict, ...]:
        # make all experiences in the TD learning form
        curr_state, next_state = {}, {}
        for key, val in experience_dict.items():
            curr_state[key] = val[:-1]
            next_state[key] = val[1:]
        return curr_state, next_state
