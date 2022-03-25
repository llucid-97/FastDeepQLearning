# Std lib
import os, pickle, itertools, typing as T, logging, copy
from collections import OrderedDict
from pathlib import Path
from threading import Thread
from queue import Queue

# 3rd party
import torch
from torch import nn, multiprocessing as mp, Tensor
from torch.utils.tensorboard import SummaryWriter

# locals
from .utils.common import soft_update, hard_update
from .components import soft_actor_critic, encoder
from franQ.Replay.wrappers import TorchDataLoader
from franQ.Agent.conf import AgentConf, AttrDict
from franQ.common_utils import PyjionJit

TensorDict = T.Dict[str, Tensor]


class DeepQLearning(nn.Module):
    def __init__(self, conf: AgentConf, **kwargs):
        nn.Module.__init__(self)

        conf: AgentConf = conf if isinstance(conf, AttrDict) else AttrDict(conf)
        self.conf = conf
        self.param_queue = kwargs.get("param_queue", mp.Queue(maxsize=1))

        # Logging
        self.summary_writer = SummaryWriter(str(Path(conf.log_dir) / f"Agent_{os.getpid()}"))

        self._define_model()

        if kwargs.get("train_process", False):
            try:
                self._initialize_trainer_members(kwargs["replays"])
                self._infinite_loop_for_async_training_process()
            except Exception as e:
                import traceback, warnings
                traceback.print_exc()
                warnings.warn("[Trainer Crashed]")

    def _define_model(self):
        conf = self.conf
        self.fast_params = []
        self.param_dict = {}
        # Models

        # Default encoder type
        self.encoder = encoder.Encoder(conf.obs_space, conf.latent_state_dim, conf.encoder_conf)
        self.fast_params += list(self.encoder.parameters())
        self.param_dict["encoder"] = self.encoder.param_dict

        if conf.use_distributional_sac:
            from .components.distributional_soft_actor_critic import DistributionalSoftActorCritic
            self.actor_critic = DistributionalSoftActorCritic(conf, conf.latent_state_dim)
        else:
            self.actor_critic = soft_actor_critic.SoftActorCritic(conf, conf.latent_state_dim)
        self.fast_params += list(self.actor_critic.parameters())
        self.param_dict["actor_critic"] = self.actor_critic.param_dict

    def enable_training(self, replays):
        conf = self.conf
        if conf.use_async_train:
            self._launch_async_training(replays)
        else:
            # Sync mode means rollout and trainer are in same process, so must be on same device
            conf.training_device = conf.inference_device
            self._initialize_trainer_members(replays)

    def _launch_async_training(self, replays):
        # make another process for doing parameter updates asynchronously
        conf = self.conf
        mp.Process(target=DeepQLearning, args=(conf,),
                   kwargs={"param_queue": self.param_queue,
                           "train_process": True,
                           "replays": replays}).start()
        # grab updated params from the other process
        Thread(target=self._pull_params).start()

    def _infinite_loop_for_async_training_process(self):
        # Sets up a thread to push newest params to inference process
        dump_q = Queue(maxsize=1)
        Thread(target=self._push_params, args=[dump_q]).start()

        # actual infinite loop[
        with PyjionJit():
            for step_train in itertools.count():
                self.train_step()
                if (step_train % self.conf.param_update_interval) == 0:
                    # Signal the _push_params thread that we've updated enough times to warrant a push
                    if dump_q.empty(): dump_q.put_nowait(None)

    def _initialize_trainer_members(self, replays):
        conf = self.conf
        self.replays = [TorchDataLoader(r, conf.training_device, conf.dtype) for r in replays]
        self.to(conf.training_device)
        self.optimizers = [torch.optim.Adam(
            self.parameters(),
            lr=self.conf.learning_rate
        )]

    def train_step(self):
        for replay in self.replays:
            experience_dict = replay.temporal_sample()
            task_loss = self.get_losses(experience_dict)

            [o.zero_grad() for o in self.optimizers]
            task_loss.backward()

            # More expensive logging should happen less frequently
            if (self.conf.global_step.value % (self.conf.log_interval * 4)) == 0:
                # Gradient norms
                for k in self.param_dict:
                    for k2 in self.param_dict[k]:
                        self.summary_writer.add_scalars(
                            f"GradNorms/{k}_{k2}",
                            {str(i): p.grad.norm() for i, p in enumerate(self.param_dict[k][k2])},
                            self.conf.global_step.value
                        )
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
        with PyjionJit():
            while True:
                _ = q.get()
                state_dict = self.state_dict()
                state_dict = OrderedDict({k: v.to("cpu:0") for k, v in state_dict.items()})
                self.param_queue.put(state_dict)

    def _pull_params(self):
        with PyjionJit():
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
            encodings, hidden_state = self.encoder.forward_eval(experiences)
            explore_action, log_prob, exploit_action = self.actor_critic.act(encodings)
            exploit_mask = experiences["exploit_mask"]
            action = (exploit_action * exploit_mask) + (explore_action * torch.logical_not(exploit_mask))

            # if self.conf.discrete: action = action.argmax(-1, True)  # go from one-hot encoding to sparse
            return action, hidden_state

    def get_random_hidden(self):
        return self.encoder.get_random_hidden()

    def reset(self):
        self.encoder.reset()

    def update_targets(self):
        self.actor_critic.update_target()

    def get_losses(self, xp: TensorDict):
        conf = self.conf
        seq_dim, batch_dim, *_ = xp["task_done"].shape
        xp["mask"] = torch.logical_not(xp["task_done"])
        xp["is_contiguous"] = xp["episode_step"][1:] == (xp["episode_step"][:-1] + 1)
        xp["is_contiguous"] *= xp["mask"][:-1]
        with torch.no_grad():
            # Convert discrete actions to 1-hot encoding
            if conf.discrete:
                xp["action_onehot"] = torch.eye(
                    conf.action_space.n,
                    device=xp["action"].device, dtype=xp["action"].dtype
                )[xp["action"].view(xp["action"].shape[:-1]).long()]

        # Run on all then split
        xp["state"] = self.encoder.forward_train(xp)
        curr_xp, next_xp = self._temporal_difference_shift(xp)

        q_loss, bootstrapped_lowerbound_loss, q_summaries = self.actor_critic.q_loss(curr_xp, next_xp)
        pi_loss, alpha_loss, pi_summaries = self.actor_critic.actor_loss(curr_xp)

        if conf.encoder_conf.use_burn_in:
            xp["is_contiguous"][:int(seq_dim * conf.encoder_conf.burn_in_portion)] = 0

        loss = ((q_loss + pi_loss + alpha_loss) * xp["is_contiguous"]) # Once its recurrent, they all use TD
        loss = loss.sum(0) / (xp["is_contiguous"].float().sum(0) +1e-4) # for RNNs we need to normalize by sequence length else network is biased
        loss = loss.mean()
        if self.conf.use_bootstrap_minibatch_nstep:
            bootstrapped_lowerbound_loss = (bootstrapped_lowerbound_loss * xp["is_contiguous"].prod(0)).mean()
            loss = loss + bootstrapped_lowerbound_loss

        # Logging metrics && debug info to tensorboard
        step = self.conf.global_step.value
        if (step % self.conf.log_interval) == 0:
            assert q_loss.shape == pi_loss.shape == xp["is_contiguous"].shape == alpha_loss.shape, \
                f"loss shape mismatch: q={q_loss.shape} pi={pi_loss.shape} c={xp['is_contiguous'].shape} a={alpha_loss.shape}"
            self.summary_writer.add_scalars("Trainer/RL_Loss", {"Critic": q_loss.mean().item(),
                                                                "Actor": pi_loss.mean().item(),
                                                                "Alpha": alpha_loss.mean().item(), },
                                            step)

            [self.summary_writer.add_scalar(f"Trainer/Critic_{k}", v, step) for k, v in q_summaries.items()]
            [self.summary_writer.add_scalar(f"Trainer/Actor_{k}", v, step) for k, v in pi_summaries.items()]
            self.summary_writer.add_scalars(
                f"Trainer/Valid_Portion",
                {"mean": xp["is_contiguous"].float().mean(),
                 "max": xp["is_contiguous"].float().sum(axis=0).max(axis=0)[0] / self.conf.temporal_len,
                 "min": xp["is_contiguous"].float().sum(axis=0).min(axis=0)[0] / self.conf.temporal_len,
                 }, step)



        return loss / self.conf.temporal_len

    @staticmethod
    def _temporal_difference_shift(experience_dict: TensorDict) -> T.Tuple[TensorDict, ...]:
        # make all experiences in the TD learning form
        curr_state, next_state = {}, {}
        for key, val in experience_dict.items():
            curr_state[key] = val[:-1]
            next_state[key] = val[1:]
        return curr_state, next_state

    def save(self, logdir):
        logdir = Path(logdir)
        logdir.mkdir(parents=True, exist_ok=True)

        conf = copy.copy(self.conf)
        conf.global_step = conf.global_step.value
        torch.save(conf, logdir / "conf.tch")
        torch.save(self.state_dict(), logdir / "state_dict.tch")

    @staticmethod
    def load_from_file(logdir):
        logdir = Path(logdir)

        conf = torch.load(logdir / "conf.tch")

        conf.global_step = mp.Value("i", conf.global_step)
        state_dict = torch.load(logdir / "state_dict.tch")
        agent = DeepQLearning(conf)
        agent.load_state_dict(state_dict)
        return agent
