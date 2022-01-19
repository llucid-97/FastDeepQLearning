from torch import multiprocessing as mp
from franQ.common_utils import AttrDict
import typing as T


class MultiProcessingWrapper():
    """Launch a new environment in another process and provide a wrapper API for it"""

    def __init__(self, env_maker, kwargs: AttrDict):
        self._queues = {
            "command": mp.Queue(maxsize=1),
            "response": mp.Queue(maxsize=1)
        }
        self._proc = mp.Process(target=ChildProcess, args=[env_maker, kwargs.to_dict(), self._queues])
        self._proc.start()

        self.action_space = self._get_proc_attr("action_space")
        self.observation_space = self._get_proc_attr("observation_space")
        self.reward_range = self._get_proc_attr("reward_range")
        self.metadata = self._get_proc_attr("metadata")

    def _get_proc_attr(self, attr):
        self._queues["command"].put(("getattr", attr))
        return self._queues["response"].get()

    def __getattr__(self, name):
        if "_proc" in self.__dict__:
            return self._get_proc_attr(name)

    def step(self, action):
        self._queues["command"].put(("step", action))
        return self._queues["response"].get()

    def reset(self, **kwargs):
        self._queues["command"].put(("reset", kwargs))
        return self._queues["response"].get()

    def render(self, mode='human', **kwargs):
        kwargs["mode"] = mode
        self._queues["command"].put(("render", kwargs))
        return self._queues["response"].get()

    def close(self):
        self._queues["command"].put(("close", None))
        return self._queues["response"].get()

    def seed(self, seed=None):
        self._queues["command"].put(("seed", seed))
        return self._queues["response"].get()

    @property
    def unwrapped(self):
        raise ValueError("Cannot return unwrapped env as it is in another process. Use get_unwrapped_attr [TODO]")


class ChildProcess():
    def __init__(self, env_maker, kwargs, queues: T.Dict[str, mp.Queue]):
        # makes the env in the new process and parses communications with the host process
        self.env = env_maker(AttrDict().from_dict(kwargs))
        self.queues = queues

        self.loop()

    def __getattr__(self, name):
        if "env" in self.__dict__:
            return getattr(self.env, name)

    def getattr(self, name):
        return getattr(self, name)

    def step(self, action):
        return self.env.step(action)

    def reset(self, kwargs):
        return self.env.reset(**kwargs)

    def render(self, kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def loop(self):
        while True:
            try:
                command, args = self.queues["command"].get()
                if command == "close":
                    self.close()
                    self.queues["response"].put(None)
                    exit(0)
                else:
                    response = getattr(self, command)(args)
                    self.queues["response"].put(response)
            except Exception as e:
                import traceback, logging
                traceback.print_exc()
                logging.error(e)
