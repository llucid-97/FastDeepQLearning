import random, shutil, uuid, itertools, typing as T, copy
from threading import Thread
from queue import Queue
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from franQ import Env, Agent
from franQ.common_utils import TimerTB
from .env_handler import env_handler
from .runner import Runner


class Evaluator(Runner):
    def __init__(self, conf: T.Union[Agent.AgentConf, Env.EnvConf], **kwargs):
        self.conf = conf
        TimerTB.CLASS_ENABLE_SWITCH = conf.enable_timers
        self.make_queues()
        self._queue_to_replay_handler = None
        # make agent: It is responsible for its own training asynchronously, and provides an API for inference
        from franQ.Agent.deepQlearning import DeepQLearning
        self.agent: DeepQLearning = Agent.make(conf)
        self.agent.to(conf.inference_device)
        self._setup_pipeline_threads()

    def _setup_pipeline_threads(self):
        """launch thread workers for each stage of the pipeline"""
        threads = []
        threads += [Thread(target=self._env_dataloader),
                    Thread(target=self._agent_dataloader),
                    Thread(target=self._agent_handler)]
        self.pipeline_threads = threads
        [t.start() for t in threads]

    def eval_agent(self, agent_path, num_episodes, seeds):
        print(f"evaluationg agent {agent_path}")
        assert len(seeds)==self.conf.num_instances
        self.agent = self.agent.load_from_file(agent_path)
        self.agent.to(self.conf.inference_device)

        self.conf.eval_envs = list(range(len(seeds)))
        threads = [Thread(
            target=env_handler,
            args=(self.conf, i, self._queue_env_handler_to_agent_dataloader, self._queue_to_environment_handler[i],
                  self._queue_to_ranker,),
            kwargs={'num_episodes': num_episodes, "seed": seeds[i], "wait_for_ranker": True}
        ) for i in range(len(seeds))]

        scores = []
        kill_switch = False

        def accumulate_scores():
            while not kill_switch:
                s = self._queue_to_ranker.get()
                if s is not None:
                    scores.append(s)

        accumulator_thread = Thread(target=accumulate_scores)
        accumulator_thread.start()
        [t.start() for t in threads]
        [t.join() for t in threads]
        self._queue_to_ranker.put(None)
        kill_switch = True
        accumulator_thread.join()

        return scores


