import typing as T
from threading import Thread

from franQ.common_utils import TimerTB
from .env_handler import env_handler
from .runner import Runner
from pathlib import Path
import math


class Evaluator(Runner):
    def __init__(self):
        pass

    def __call__(self,
                 agent_path: T.Union[str, Path],
                 num_episodes,
                 parallel_instance_seeds: T.Union[int, T.List[int]] = None,
                 env_generator: T.Callable = None,
                 log_dir_override=None,
                 deterministic=True,
                 render=True,
                 device='cpu',
                 preproc=False):
        print(f"evaluationg agent {agent_path}")
        agent_path = Path(agent_path)
        assert agent_path.exists()
        from franQ.Agent.deepQlearning import DeepQLearning

        self.agent = DeepQLearning.load_from_file(agent_path)
        self.agent.to(device)

        self.conf = conf = self.agent.conf
        conf.render = render
        if log_dir_override is not None:
            self.conf.log_dir = log_dir_override
        else:
            import tempfile
            from franQ import common_utils
            self.conf.log_dir = Path(tempfile.gettempdir()) / common_utils.time_stamp_str()

        TimerTB.CLASS_ENABLE_SWITCH = conf.enable_timers
        try:
            parallel_instance_seeds = list(parallel_instance_seeds)
        except TypeError:
            parallel_instance_seeds = [parallel_instance_seeds]
        self.conf.num_instances = len(parallel_instance_seeds)

        if deterministic:
            self.conf.eval_envs = list(range(len(parallel_instance_seeds)))
        else:
            self.conf.eval_envs = []

        self.make_queues()
        self._setup_pipeline_threads()

        self._queue_to_replay_handler = None

        threads = [Thread(
            target=env_handler,
            args=(self.conf, i, self._queue_env_handler_to_agent_dataloader, self._queue_to_environment_handler[i],
                  self._queue_to_ranker,),
            kwargs={'num_episodes': int(math.ceil(num_episodes / len(parallel_instance_seeds))),
                    "seed": parallel_instance_seeds[i],
                    "wait_for_ranker": True,
                    "env_generator": env_generator,
                    "preproc":preproc,}
        ) for i in range(len(parallel_instance_seeds))]

        scores = []

        def accumulate_scores():
            while (data:= self._queue_to_ranker.get()) is not None:
                scores.append(data)

        accumulator_thread = Thread(target=accumulate_scores)
        accumulator_thread.start()
        [t.start() for t in threads]
        [t.join() for idx,t in enumerate(threads)]
        self._queue_to_ranker.put(None)
        accumulator_thread.join()
        return scores

    def _setup_pipeline_threads(self):
        """launch thread workers for each stage of the pipeline"""
        threads = []
        threads += [Thread(target=self._env_dataloader),
                    Thread(target=self._agent_dataloader),
                    Thread(target=self._agent_handler)]
        self.pipeline_threads = threads
        [t.start() for t in threads]
