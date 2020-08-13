import torch
from torch import nn, Tensor, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from .AlgoModules.SAC_Module import sac_baseline, sac_sde
from Agent.conf import AgentConf

import typing as T
from threading import Thread
from queue import Queue

from Replay.async_replay_memory import AsyncReplayMemory
from Replay.wrappers.torch_dataloader import TorchDataLoader

from collections import OrderedDict
import traceback, itertools
from pathlib import Path


def make_module(conf) -> nn.Module:
    if conf.use_sde:
        Module = sac_sde.SDESoftActorCriticModule
    else:
        Module = sac_baseline.SoftActorCriticModule
    return Module(conf, conf.obs_space.shape[-1])


class SoftActorCriticAgent(nn.Module):
    def __init__(self, conf: AgentConf, replays: T.List[AsyncReplayMemory]):
        nn.Module.__init__(self)
        self.sac = make_module(conf)
        self.conf = conf
        from gym.spaces import Discrete
        self.discrete = isinstance(conf.action_space, Discrete)
        self.step = 0
        if conf.use_async_train:
            # make another process for doing parameter updates asynchronously and make a queue to get params from it
            self.param_q = mp.Queue(maxsize=1)
            mp.Process(target=train_sac, args=[conf.to_dict(), replays, self.param_q]).start()
            Thread(target=self._update_params).start()
        else:
            conf.training_device = conf.inference_device # override since they now share a model
            self.optimizer, self.logger, self.replays = setup_trainer_components(conf, self.sac, replays)

    @property
    def iteration(self):
        return self.step

    def _update_params(self):
        while True:
            self.step, state_dict = self.param_q.get()
            self.sac.load_state_dict(state_dict)

    def act(self, experiences: T.Dict[str, Tensor]) -> T.Dict[str, Tensor]:
        """Run agent inference. If async mode is disabled, take a training step too"""
        if not self.conf.use_async_train:
            if self.replays[0].ready():
                train_step(self.step, self.conf, self.replays, self.sac, self.optimizer, self.logger)
                self.step += 1

        with torch.no_grad():
            rsampled, log_prob, mean = self.sac.act(experiences["obs"])

            if self.conf.num_instances > 1:
                # Vectorized Choose: whether to explore or exploit
                exploit_mask = (experiences["idx"] == 0).view(-1, 1)
                action = (mean * exploit_mask) + (rsampled * torch.logical_not(exploit_mask))
            else:
                action = rsampled # explore always if we only have 1 environment

            if self.discrete: action = action.argmax(-1, True)  # go from one-hot encoding to sparse
            return action


def setup_trainer_components(conf: AgentConf, sac: nn.Module, replays: T.List[AsyncReplayMemory]):
    optimizer = torch.optim.Adam(sac.parameters(), lr=conf.lr)
    logger = SummaryWriter(Path(conf.log_dir) / "Trainer")
    replays = [TorchDataLoader(r, conf.training_device, conf.dtype) for r in replays]
    return optimizer, logger, replays


def train_sac(conf: AgentConf, replays, param_q: mp.Queue):
    try:
        conf = AgentConf().from_dict(conf)
        sac = make_module(conf)
        sac.to(conf.training_device)
        optimizer, logger, replays = setup_trainer_components(conf, sac, replays)

        # Parallelize parameter dump to CPU
        def dump_params(dump_q: Queue):
            while True:
                step, state_dict = dump_q.get()
                state_dict = OrderedDict({k: v.to("cpu:0") for k, v in state_dict.items()})
                param_q.put((step, state_dict))

        dump_q = Queue(maxsize=1)
        Thread(target=dump_params, args=[dump_q]).start()

        # Perform training steps
        for step in itertools.count():
            train_step(step, conf, replays, sac, optimizer, logger)
            if (step % conf.dump_period) == 0:
                # Push params to experiment runner
                if dump_q.empty():
                    dump_q.put_nowait((step, sac.state_dict()))

    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
        print()


def train_step(step, conf: AgentConf, replays: T.List[TorchDataLoader], sac: nn.Module, optimizer: torch.optim.Adam,
               logger: SummaryWriter):
    num_replays = len(replays)
    experience: T.Dict[str, Tensor] = replays[step % num_replays].temporal_sample(conf.batch_size,
                                                                                  conf.temporal_len)

    # for compatibility with other usages of the sac module, we must comply with the names it expects!
    experience["state"] = experience["obs"]
    experience["mask"] = torch.logical_not(experience["task_done"])

    # Ensure scalars have a separate dimension
    curr_xp, next_xp = {}, {}
    for k in experience:
        if len(experience[k].shape) == 2:
            experience[k].unsqueeze_(-1)

        # ensure we have a temporal-difference style.
        curr_xp[k] = experience[k][:-1]
        next_xp[k] = experience[k][1:]

    # Calculate loss
    critic_loss, bootstrap_nstep_loss, critic_summaries = sac.critic_loss(curr_xp, next_xp)
    actor_loss, entropy_loss, actor_summaries = sac.actor_loss(curr_xp)

    # Combine losses and Mask the invalid (non-contiguous) elements
    contiguous_mask = next_xp["ep_step"] == (curr_xp["ep_step"] + 1)
    loss = critic_loss + actor_loss + entropy_loss
    loss = (loss * contiguous_mask).mean()
    if bootstrap_nstep_loss is not None:
        bootstrap_nstep_loss = (bootstrap_nstep_loss * contiguous_mask.prod(0)).mean()
        loss = loss + bootstrap_nstep_loss

    loss = loss / conf.temporal_len

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(sac.parameters(), max_norm=conf.grad_clip)
    optimizer.step()
    sac.update_target()

    if (step % conf.dump_period) == 0:
        # Push logs to disk
        logger.add_scalar("loss/Critic", critic_loss[0][0][0], step)
        logger.add_scalar("loss/Actor", actor_loss[0][0][0], step)
        logger.add_scalar("loss/Entropy", entropy_loss[0][0][0], step)

        [logger.add_scalar(f"Critic/{k}", v, step) for k, v in critic_summaries.items()]
        [logger.add_scalar(f"Actor/{k}", v, step) for k, v in actor_summaries.items()]
