import random, shutil, uuid, itertools, typing as T, copy
from threading import Thread
from queue import Queue
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from franQ import Env, Replay, Agent
from franQ.common_utils import TimerTB, PyjionJit
from .env_handler import env_handler

class Runner:
    """Manages the experiment: Handles interactions between agent, environment and replay memory"""

    def __init__(self, conf: T.Union[Agent.AgentConf, Env.EnvConf], **kwargs):
        self.conf = conf
        TimerTB.CLASS_ENABLE_SWITCH = conf.enable_timers
        self.make_queues()

        # shards are separate replay memories for each env-actor pair so we don't violate FIFO assumptions for storage
        shards, self.replay_shards = Replay.make(conf, **kwargs)

        # make agent: It is responsible for its own training asynchronously, and provides an API for inference
        from franQ.Agent.deepQlearning import DeepQLearning
        self.agent: DeepQLearning = Agent.make(conf)
        self.agent.enable_training(shards)
        self.agent.to(conf.inference_device)

    def make_queues(self):
        conf = self.conf
        # Queues for communicating between all worker threads
        self._queue_env_handler_to_agent_dataloader = Queue(maxsize=conf.num_instances)
        self._queue_agent_dataloader_to_agent_handler = Queue(maxsize=1)
        self._queue_agent_handler_to_env_dataloader = Queue(maxsize=1)
        self._queue_to_environment_handler = [Queue(maxsize=1) for _ in range(conf.num_instances)]
        self._queue_to_replay_handler = [Queue(maxsize=1) for _ in range(conf.num_instances)]
        self._queue_to_ranker = Queue(maxsize=1)

    def launch(self):
        """launch thread workers for each stage of the pipeline"""
        threads = [Thread(
            target=env_handler,
            args=(self.conf, i, self._queue_env_handler_to_agent_dataloader, self._queue_to_environment_handler[i],
                  (self._queue_to_ranker if i == 0 else None)
                  )
        ) for i in range(self.conf.num_instances)]
        threads += [Thread(target=self._replay_handler, args=[i]) for i in range(self.conf.num_instances)]
        threads += [Thread(target=self._env_dataloader),
                    Thread(target=self._agent_dataloader),
                    Thread(target=self._agent_handler)]
        threads += [Thread(target=self._ranker)]

        [t.start() for t in threads]
        [t.join() for t in threads]

    def _agent_dataloader(self):
        """Pipeline Stage: Asynchronously converts data to right format for agent and loads to agent's device"""
        with PyjionJit():
            logger = SummaryWriter(Path(self.conf.log_dir) / "Runner_DataLoader")
            inference_keys = None
            for step in itertools.count():
                # wait for a request to be fed recieved
                requests: T.List[dict] = [self._queue_env_handler_to_agent_dataloader.get()]

                with TimerTB(logger, "_agent_dataloader", group="timers/runner", step=step):
                    # get the rest that are already present (if any)
                    while not self._queue_env_handler_to_agent_dataloader.empty():
                        requests.append(self._queue_env_handler_to_agent_dataloader.get())

                    if inference_keys is None:
                        # Select ONLY the keys needed for agent inference. They are the same each cycle.
                        inference_keys = [k for k in self.conf.inference_input_keys if k in requests[0]]

                    # Batch the data for inference & copy to inference_device
                    batch: T.Dict[str, Tensor] = {
                        k: torch.stack([torch.tensor(xp[k], dtype=self.conf.dtype, device=self.conf.inference_device)
                                        for xp in requests])
                        for k in inference_keys
                    }
                    # bool tensor to say whether to explore/exploit for each env
                    exploit = np.isin(batch["idx"].cpu().numpy(), self.conf.eval_envs)
                    exploit = torch.from_numpy(exploit).view(-1, 1)
                    exploit.to(self.conf.inference_device)
                    batch["exploit_mask"] = exploit.view(-1, 1)

                    self._queue_agent_dataloader_to_agent_handler.put((batch, requests))

    def _agent_handler(self):
        """Pipeline Stage: Asynchronously runs agent inference on batch of env requests (observations)"""
        with PyjionJit():
            logger = SummaryWriter(Path(self.conf.log_dir) / "Runner_Inference")
            for step in itertools.count():
                batch, xp_dict_list = self._queue_agent_dataloader_to_agent_handler.get()

                with TimerTB(logger, "_agent_handler", group="timers/runner", step=step):
                    actions = self.agent.act(batch)
                    self._queue_agent_handler_to_env_dataloader.put((actions, xp_dict_list))

    def _env_dataloader(self):
        """Pipeline Stage: Asynchronously copies action from Inference_device to cpu and sends it to env and replay"""
        with PyjionJit():
            logger = SummaryWriter(Path(self.conf.log_dir) / "Runner_DataUnloader")
            for step in itertools.count():
                actions, xp_dict_list = self._queue_agent_handler_to_env_dataloader.get()
                with TimerTB(logger, "_env_dataloader", group="timers/runner", step=step):
                    actions: Tensor = actions.cpu().numpy()

                    assert actions.shape[0] == len(xp_dict_list)
                    # logger.add_scalar("Env/inference_batch_size", len(xp_dict_list), step)

                    for i in range(len(xp_dict_list)):
                        # route responses to experience dicts for right envs in list of exp dicts
                        action = actions[i]
                        env_idx = xp_dict_list[i]["idx"]
                        xp_dict_list[i]["action"] = action

                        # push experience dicts to appropriate replay memories.
                        if self._queue_to_replay_handler is not None:
                            replay_copy = copy.deepcopy(
                                xp_dict_list[i])  # copy to ensure no mutation as it is passed between threads
                            self._queue_to_replay_handler[env_idx].put(replay_copy)

                        # push actions to env
                        if self.conf.discrete: action = action.item()
                        self._queue_to_environment_handler[env_idx].put(action)

    def _replay_handler(self, idx):
        """Pipeline Stage: Asynchronously performs transformations before storing to replay.
        Transformations must be defined as replay wrappers in init"""

        logger = SummaryWriter(Path(self.conf.log_dir) / f"Runner_replay_{idx}")
        for step in itertools.count():
            experience_dict: dict = self._queue_to_replay_handler[idx].get()
            if not self.conf.use_HER:
                del experience_dict["info"]

            with TimerTB(logger, f"ReplayTransforms_{idx}", group="timers/runner", step=step):
                self.replay_shards[idx].add(experience_dict)

    def _ranker(self, leaderboard_size=10):
        with PyjionJit():
            leaderboard = []
            metadata = {}
            for _ in itertools.count():
                score = self._queue_to_ranker.get()
                if (score > np.asarray(leaderboard)).any() or len(leaderboard) == 0:
                    # Put agent into leaderboard
                    while score in metadata: score = score + ((random.random() - 0.5) * 1e-6)
                    leaderboard.append(score)
                    leaderboard = list(sorted(leaderboard, reverse=True))
                    metadata[score] = {
                        "path": Path(self.conf.log_dir) / "models" / f"score={score}_step={self.conf.global_step.value}"
                    }
                    self.agent.save(metadata[score]["path"])
                    if len(leaderboard) > leaderboard_size:
                        cull = leaderboard[leaderboard_size:]
                        leaderboard = leaderboard[:leaderboard_size]

                        for c in cull:
                            p = metadata[c]["path"]
                            if p.exists():
                                shutil.rmtree(metadata[c]["path"], ignore_errors=True)
                            del metadata[c]

                    lstring = '\n'.join([f'{i} : {l:.2f} @ step ({self.conf.global_step.value})' for i, l in enumerate(leaderboard)])
                    print(f"Top {leaderboard_size} scores: [\n{lstring}\n]")
