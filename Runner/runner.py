from threading import Thread
from queue import Queue
import Agent, Env, Replay
import torch, numpy as np
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import typing as T
from pathlib import Path
import common_utils, itertools
from common_utils import TimerSummary
import copy


class Runner:
    """Manages the experiment: Handles interactions between agent, environment and replay memory"""

    def __init__(self, conf: T.Union[Agent.AgentConf, Env.EnvConf], **kwargs):
        self.conf = conf
        common_utils.TimerSummary.CLASS_ENABLE_SWITCH = conf.enable_timers
        self.discrete = conf.discrete

        # Queues for communicating between all worker threads
        self._queue_env_handler_to_agent_dataloader = Queue(maxsize=conf.num_instances)
        self._queue_agent_dataloader_to_agent_handler = Queue(maxsize=1)
        self._queue_agent_handler_to_env_dataloader = Queue(maxsize=1)
        self._queue_to_environment_handler = [Queue(maxsize=1) for _ in range(conf.num_instances)]
        self._queue_to_replay_handler = [Queue(maxsize=1) for _ in range(conf.num_instances)]

        # shards are separate replay memories for each env-actor pair so we don't violate FIFO assumptions for storage
        shards, self.replay_shards = _make_replay_shards(conf, **kwargs)

        # make agent: It is responsible for its own training asynchronously, and provides an API for inference
        self.agent = Agent.make(conf, shards)
        self.agent.to(conf.inference_device)

    def launch(self):
        """launch thread workers for each stage of the pipeline"""
        threads = [Thread(target=self._env_handler, args=[i]) for i in range(self.conf.num_instances)]
        threads += [Thread(target=self._replay_handler, args=[i]) for i in range(self.conf.num_instances)]
        threads += [Thread(target=self._env_dataloader),
                    Thread(target=self._agent_dataloader),
                    Thread(target=self._agent_handler)]

        [t.start() for t in threads]
        [t.join() for t in threads]

    def _agent_dataloader(self):
        """Pipeline Stage: Asynchronously converts data to right format for agent and loads to agent's device"""
        logger = SummaryWriter(Path(self.conf.log_dir) / "Runner_DataLoader")
        inference_keys = None
        for step in itertools.count():
            # wait for a request to be fed recieved
            requests: T.List[dict] = [self._queue_env_handler_to_agent_dataloader.get()]

            with common_utils.TimerSummary(logger, "_agent_dataloader", group="timers/runner", step=step):
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

                self._queue_agent_dataloader_to_agent_handler.put((batch, requests))

    def _agent_handler(self):
        """Pipeline Stage: Asynchronously runs agent inference on batch of env requests (observations)"""
        logger = SummaryWriter(Path(self.conf.log_dir) / "Runner_Inference")
        for step in itertools.count():
            batch, xp_dict_list = self._queue_agent_dataloader_to_agent_handler.get()

            with TimerSummary(logger, "_agent_handler", group="timers/runner", step=step):
                actions = self.agent.act(batch)
                self._queue_agent_handler_to_env_dataloader.put((actions, xp_dict_list))

    def _env_dataloader(self):
        """Pipeline Stage: Asynchronously copies action from Inference_device to cpu and sends it to env and replay"""
        logger = SummaryWriter(Path(self.conf.log_dir) / "Runner_DataUnloader")
        for step in itertools.count():
            actions, xp_dict_list = self._queue_agent_handler_to_env_dataloader.get()
            with TimerSummary(logger, "_env_dataloader", group="timers/runner", step=step):
                actions: Tensor = actions.cpu().numpy()

                assert actions.shape[0] == len(xp_dict_list)
                # logger.add_scalar("Env/inference_batch_size", len(xp_dict_list), step)

                for i in range(len(xp_dict_list)):
                    # route responses to experience dicts for right envs in list of exp dicts
                    action = actions[i]
                    env_idx = xp_dict_list[i]["idx"]
                    xp_dict_list[i]["action"] = action

                    # push experience dicts to appropriate replay memories.
                    replay_copy = copy.deepcopy(
                        xp_dict_list[i])  # copy to ensure no mutation as it is passed between threads
                    self._queue_to_replay_handler[env_idx].put(replay_copy)

                    # push actions to env
                    if self.discrete: action = action.item()
                    self._queue_to_environment_handler[env_idx].put(action)

    def _replay_handler(self, idx):
        """Pipeline Stage: Asynchronously performs transformations before storing to replay.
        Transformations must be defined as replay wrappers in init"""
        logger = SummaryWriter(Path(self.conf.log_dir) / f"Runner_replay_{idx}")
        for step in itertools.count():
            experience_dict = self._queue_to_replay_handler[idx].get()

            with TimerSummary(logger, f"ReplayTransforms_{idx}", group="timers/runner", step=step):
                self.replay_shards[idx].add(experience_dict)

    def _env_handler(self, idx):
        """Pipeline Stage: Asynchronously handles stepping through env to get a response"""
        env = Env.make_mp(self.conf)
        logger = SummaryWriter(Path(self.conf.log_dir) / f"Runner_Env{idx}")
        total_step = 0
        for episode in itertools.count():
            if episode > self.conf.max_num_episodes: break

            # Reset & init all data from the environment
            score = 0
            experience = {"reward": 0.0,
                          "episode_done": False,
                          "task_done": False,
                          "idx": idx}
            experience.update(env.reset())
            for experience["episode_step"] in itertools.count():
                with common_utils.TimerSummary(logger, f"Pipeline_Stall{idx}", group="timers/pipeline_stats",
                                               step=total_step):
                    # Get action form agent
                    self._queue_env_handler_to_agent_dataloader.put(experience)
                    action = self._queue_to_environment_handler[idx].get()

                if experience["episode_done"]:
                    break

                with common_utils.TimerSummary(logger, f"Env_{idx}_Step", group="timers/runner_pipeline",
                                               step=total_step):
                    # Get new experience from environment and populate the dict
                    obs, experience["reward"], experience["episode_done"], info = env.step(action)
                    experience.update(obs)
                    experience["task_done"] = experience["episode_done"] and not info.get('TimeLimit.truncated', False)
                    if self.conf.render:
                        env.render()

                    score += experience["reward"]
                    total_step += 1

            logger.add_scalar("Env/Episode_Score", score, episode)
            logger.add_scalar("Env/TrainStep_Score", score, self.agent.iteration)
            # logger.add_scalar("Env/EnvStep_Score", score, total_step)


def _make_replay_shards(conf: Agent.AgentConf, **kwargs) -> T.Tuple[T.List[Replay.AsyncReplayMemory]]:
    """Helper function to construct the replay memories"""
    # Reader Shards
    shards = [Replay.AsyncReplayMemory(
        conf.replay_size, conf.batch_size, conf.temporal_len
    ) for _ in range(conf.num_instances)]

    # Construct optional wrappers around it for modular transformations
    writer_wrappers = shards
    if conf.use_squashed_rewards:
        writer_wrappers = [Replay.wrappers.squash_rewards.SquashRewards(r) for r in writer_wrappers]
    if conf.use_HER:
        writer_wrappers = [Replay.wrappers.her.HindsightNStepReplay(r, conf.nStep_return_steps, conf.gamma,
                                                                    kwargs["compute_reward"]) for r in writer_wrappers]
    if conf.use_nStep_lowerbounds and not conf.use_HER:
        writer_wrappers = [Replay.wrappers.nstep_return.NStepReturn(r, conf.nStep_return_steps, conf.gamma) for r in
                           writer_wrappers]
    return shards, writer_wrappers
