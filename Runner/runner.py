from threading import Thread
from queue import Queue
import Agent, Env, Replay
import torch, numpy as np
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import typing as T
from pathlib import Path
import common_utils, itertools


class Runner:
    """Manages the experiment: Handles interactions between agent, environment and replay memory"""

    def __init__(self, conf: T.Union[Agent.AgentConf, Env.EnvConf]):
        self.conf = conf
        common_utils.TimerSummary.CLASS_ENABLE_SWITCH = conf.enable_profiling
        from gym.spaces import Discrete
        self.discrete = isinstance(self.conf.action_space,Discrete)

        # Queues for communicating between all worker threads
        self.obs_queue = Queue(maxsize=conf.num_instances)
        self.agent_fetch_q = Queue(maxsize=3)
        self.agent_dump_q = Queue(maxsize=3)
        self.action_queues = [Queue(maxsize=1) for _ in range(conf.num_instances)]
        self.replay_queues = [Queue(maxsize=3) for _ in range(conf.num_instances)]

        # shards are separate replay memories for each env-actor pair so we don't violate FIFO assumptions for storage
        self.replay_shards = _make_replay_shards(conf)

        # make agent: It is responsible for its own training asynchronously, and provides an API for inference
        self.agent = Agent.make(conf, self.replay_shards)
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
        for step in itertools.count():
            # wait for a request to be fed recieved
            requests: T.List[dict] = [self.obs_queue.get()]  # make it stall & wait for the first one

            # get all that are present to process as a batch
            with common_utils.TimerSummary(logger, "Inference_DataLoader", group="timers/runner_pipeline", step=step):
                while not self.obs_queue.empty():
                    requests.append(self.obs_queue.get())

                # Convert inputs the agent needs into batched tensors & push to device
                batch: T.Dict[str, Tensor] = {}
                for k in self.conf.inference_input_keys:
                    batch[k] = torch.stack([torch.tensor(xp[k], dtype=self.conf.dtype, device=self.conf.inference_device)
                                            for xp in requests])

                # Push to inference
                # TODO: insert callback for performing transformations here (eg transpose for images)
                self.agent_fetch_q.put((batch, requests))  # serve to agent

    def _agent_handler(self):
        """Pipeline Stage: Asynchronously runs agent inference on batch of env requests (observations)"""
        logger = SummaryWriter(Path(self.conf.log_dir) / "Runner_Inference")
        for step in itertools.count():
            batch, xp_dict_list = self.agent_fetch_q.get()  # grab new observations from prefetcher

            with common_utils.TimerSummary(logger, "Inference", group="timers/runner_pipeline", step=step):
                actions = self.agent.act(batch)
                self.agent_dump_q.put((actions, xp_dict_list))  # pass it to env fetcher and list of dicts

    def _env_dataloader(self):
        """Pipeline Stage: Asynchronously copies action from DEVICE to cpu and sends it to env and replay"""
        logger = SummaryWriter(Path(self.conf.log_dir) / "Runner_DataUnloader")
        for step in itertools.count():
            actions, xp_dict_list = self.agent_dump_q.get()
            with common_utils.TimerSummary(logger, "Env_Dataloader", group="timers/runner_pipeline", step=step):
                actions: Tensor = actions.cpu().numpy()
                assert actions.shape[0] == len(xp_dict_list)
                # TODO: Log batch size to tensorboard
                for i in range(len(xp_dict_list)):
                    # route responses to experience dicts for right envs in list of exp dicts
                    action = actions[i]
                    env_idx = xp_dict_list[i]["idx"]
                    xp_dict_list[i]["action"] = np.copy(action)

                    # push experience dicts to appropriate replay memories.
                    replay_copy = {}  # copy so others cant mutate. Deep copy would be safer, but less efficient & unnecessary if values are not mutated in place
                    replay_copy.update(xp_dict_list[i])

                    with common_utils.TimerSummary(logger, "Env_Dataloader___REPLAY_ADD", group="timers", step=step):
                        self.replay_queues[env_idx].put(replay_copy)

                    # push actions to env
                    if self.discrete: action = action.item()
                    with common_utils.TimerSummary(logger, "Env_Dataloader___ACTIONQ", group="timers", step=step):
                        self.action_queues[env_idx].put(action)

    def _replay_handler(self,idx):
        """Pipeline Stage: Asynchronously performs transformations before storing to replay.
        Transformations must be defined as replay wrappers in init"""
        logger = SummaryWriter(Path(self.conf.log_dir) / "Runner_replay_transformer")
        for step in itertools.count():
            experience_dict = self.replay_queues[idx].get()
            with common_utils.TimerSummary(logger, "Replay Transformer", group="timers/runner_pipeline", step=step):
                self.replay_shards[idx].add(experience_dict)

    def _env_handler(self, idx):
        """Pipeline Stage: Asynchronously handles stepping through env to get a response"""
        import itertools
        env = Env.make_mp(self.conf)
        logger = SummaryWriter(Path(self.conf.log_dir) / f"Runner_Env{idx}")
        total_step = 0
        for episode in itertools.count():
            if episode > self.conf.max_num_episodes: break

            # Reset & init all data from the environment
            score = 0
            experience = {"obs": env.reset(),
                          "reward": 0.0,
                          "ep_done": False,
                          "task_done": False,
                          "idx": idx}
            for experience["ep_step"] in itertools.count():
                with common_utils.TimerSummary(logger, f"Pipeline_Stall{idx}", group="timers/runner_pipeline",
                                               step=total_step):
                    # send observation to agent
                    self.obs_queue.put(experience)
                    # get response from agent to last thing seen
                    action = self.action_queues[idx].get()

                    # May terminate here (ensure the agent gets the done observation before quitting)
                    if experience["ep_done"]:
                        logger.add_scalar("Score/Episode", score, episode)
                        logger.add_scalar("Score/TrainStep", score, self.agent.iteration)
                        break

                with common_utils.TimerSummary(logger, f"Env_{idx}_Step", group="timers/runner_pipeline",
                                               step=total_step):
                    # Get new experience from environment and populate the dict
                    experience["obs"], experience["reward"], experience["ep_done"], info = env.step(action)
                    experience["task_done"] = experience["ep_done"] and not info.get('TimeLimit.truncated', False)
                    env.render()

                    # update stats
                    score += experience["reward"]
                    total_step += 1




def _make_replay_shards(conf: Agent.AgentConf) -> T.List[Replay.AsyncReplayMemory]:
    """Helper function to construct the replay memories"""
    shards = [Replay.AsyncReplayMemory(
        conf.replay_size, conf.batch_size, conf.temporal_len
    ) for _ in range(conf.num_instances)]

    # Construct optional wrappers around it for modular transformations
    if conf.squash_rewards:
        shards = [Replay.wrappers.squash_rewards.SquashRewards(r) for r in shards]
    if conf.use_nStep_lowerbounds:
        shards = [Replay.wrappers.nstep_return.NStepReturn(r, conf.mc_return_step, conf.gamma) for r in shards]
    return shards
