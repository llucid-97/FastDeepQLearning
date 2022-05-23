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

from collections import defaultdict


class Runner:
    """Manages the experiment: Handles data pipeline between agent, environment and replay memory
    - It stages each component (env, agent, memory) in parallel and creates threads to handle each
    - It sets up queues for each to communicate with one another asynchronously
    - It sets up threads for pre-fetching data between CPU and GPU asynchronously

    Important Notes:
    - It DOES NOT handle training.
    - Logging uses a "global_step" from the config which this class IS NOT RESPONSIBLE for mutating
    (i.e. If it is not incremented externally, all logs will be tagged with the same step (0)).
    """

    def __init__(self, conf: T.Union[Agent.AgentConf, Env.EnvConf], **kwargs):
        self.conf = conf
        TimerTB.CLASS_ENABLE_SWITCH = conf.enable_timers
        self.make_queues()

        # shards are separate replay memories for each environment so we don't break FIFO assumptions for storage
        shards, self.replay_shards = Replay.make(conf, **kwargs)

        # make agent: NOTE: Agent is responsible for its own training. Only provides an API for inference here
        from franQ.Agent.deepQlearning import DeepQLearning
        self.agent: DeepQLearning = Agent.make(conf)
        self.agent.enable_training(shards)
        self.agent.to(conf.inference_device)

    def make_queues(self):
        """This sets up Queues for communicating between all worker threads"""
        conf = self.conf
        self._queue_env_handler_to_agent_dataloader = Queue(maxsize=1)
        self._queue_agent_dataloader_to_agent_handler = Queue(maxsize=1)
        self._queue_agent_handler_to_env_dataloader = Queue(maxsize=1)
        self._queue_to_environment_handler = [Queue(maxsize=1) for _ in range(conf.num_instances)]
        self._queue_to_replay_handler = [Queue(maxsize=1) for _ in range(conf.num_instances)]
        self._queue_to_ranker = Queue(maxsize=1)

    def launch(self, block=True):
        """launch thread workers for each stage of the pipeline
        """

        # Spin up a thread for every environment instance
        threads = [Thread(target=env_handler,  #
                          kwargs={
                              "conf": self.conf,
                              "idx": i,
                              "queue_put_experience": self._queue_env_handler_to_agent_dataloader,
                              "queue_get_action": self._queue_to_environment_handler[i],
                              "queue_put_score": (self._queue_to_ranker if i == 0 else None),
                          }
                          ) for i in range(self.conf.num_instances)]

        # Spin up a thread for every replay instance
        threads += [Thread(target=self._replay_handler, args=[i]) for i in range(self.conf.num_instances)]

        # Spin up thread for Agent and the dataloaders which prefetch data to and from GPU
        threads += [Thread(target=self._env_dataloader),
                    Thread(target=self._agent_dataloader),
                    Thread(target=self._agent_handler)]

        # Tracking top N agents & saves them. Deletes any that falls off leaderboard to avoid excess disk usage
        threads += [Thread(target=self._ranker)]

        [t.start() for t in threads]
        if block:
            [t.join() for t in threads]
        else:
            self.threads = threads

    def _agent_dataloader(self):
        """Pipeline Stage: prepare and prefetch data to GPU for agent"""

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

                    # Handle passing hidden state here
                    for r in requests:
                        if "agent_state" not in r:
                            agent_state = self.agent.get_random_hidden()
                            if agent_state is not None:
                                r["agent_state"] = agent_state

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
        """Pipeline Stage: Runs agent inference on batch of env requests (observations)"""
        with PyjionJit():
            logger = SummaryWriter(Path(self.conf.log_dir) / "Runner_Inference")
            for step in itertools.count():
                batch, xp_dict_list = self._queue_agent_dataloader_to_agent_handler.get()

                with TimerTB(logger, "_agent_handler", group="timers/runner", step=step):
                    actions, hidden_state, info = self.agent.act(batch)
                    self._queue_agent_handler_to_env_dataloader.put((actions, hidden_state, xp_dict_list, info))

    def _env_dataloader(self):
        """Pipeline Stage: prefetch from agent->[env, memory]"""
        with PyjionJit():
            conf = self.conf
            writer = SummaryWriter(Path(self.conf.log_dir) / "Runner_DataUnloader")
            last_logged_step = defaultdict(lambda: -conf.log_interval)
            for step in itertools.count():
                actions, hidden_state, xp_dict_list, info = self._queue_agent_handler_to_env_dataloader.get()
                curr_train_step = self.conf.train_step.value
                with TimerTB(writer, "_env_dataloader", group="timers/runner", step=step):
                    actions: Tensor = actions.cpu().numpy()
                    assert actions.shape[0] == len(xp_dict_list)
                    # logger.add_scalar("Env/inference_batch_size", len(xp_dict_list), step)
                    if hidden_state is not None:
                        hidden_state = hidden_state.cpu().numpy()
                    for i in range(len(xp_dict_list)):
                        # route responses to experience dicts for right envs in list of exp dicts
                        action = actions[i]
                        env_idx = xp_dict_list[i]["idx"]
                        xp_dict_list[i]["action"] = action
                        if hidden_state is not None:
                            xp_dict_list[i]["agent_state"] = hidden_state[i]

                        # push experience dicts to appropriate replay memories.
                        if self._queue_to_replay_handler is not None:
                            replay_copy = copy.deepcopy(xp_dict_list[i])  # ensure no mutation between threads
                            self._queue_to_replay_handler[env_idx].put(replay_copy)

                        if len(info) and (abs(last_logged_step[env_idx] - curr_train_step) > (conf.log_interval / 2)):
                            print(f"last_logged_step[{env_idx}]={last_logged_step[env_idx]} curr_train_step={curr_train_step}")
                            last_logged_step[env_idx] = curr_train_step
                            for k, v in info.items():
                                writer.add_scalars(f"inference/{k}", {str(env_idx): v[i]}, global_step=curr_train_step)

                        # push actions to env
                        if self.conf.discrete: action = action.item()
                        if hidden_state is None:
                            self._queue_to_environment_handler[env_idx].put((action, None))
                        else:
                            self._queue_to_environment_handler[env_idx].put((action, hidden_state[i]))

    def _replay_handler(self, idx):
        """Pipeline Stage: Asynchronously performs transformations before storing to replay.
        Transformations must be defined as replay wrappers in init"""

        logger = SummaryWriter(str(Path(self.conf.log_dir) / f"Runner_replay_{idx}"))

        for step in itertools.count():
            xp: dict = self._queue_to_replay_handler[idx].get()
            if not self.conf.use_HER:
                del xp["info"]

            # if "agent_state" in xp:
            #     del xp["agent_state"]
            with TimerTB(logger, f"ReplayTransforms_{idx}", group="timers/runner", step=step):
                self.replay_shards[idx].add(xp)

    def _ranker(self, leaderboard_size=10):
        with PyjionJit():
            leaderboard = []
            metadata = {}
            conf = self.conf
            for _ in itertools.count():
                data = self._queue_to_ranker.get()
                score = data["score"]
                if (score > np.asarray(leaderboard)).any() or len(leaderboard) == 0:
                    # Put agent into leaderboard
                    while score in metadata: score = score + ((random.random() - 0.5) * 1e-6)
                    leaderboard.append(score)
                    leaderboard = list(sorted(leaderboard, reverse=True))
                    metadata[score] = {
                        "path": Path(conf.log_dir) / "models" / f"score={score}_trainstep={conf.train_step.value}",
                        "name": f"score={score:.2f} train_step={conf.train_step.value} env_step={data['step']}",
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

                    lstring = '\n'.join([f'{i} : {metadata[l]["name"]})' for i, l in enumerate(leaderboard)])
                    print(f"Top {leaderboard_size} scores: [\n{lstring}\n]")
