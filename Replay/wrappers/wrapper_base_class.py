from ..replay_memory import ReplayMemory
class ReplayMemoryWrapper:
    """
    Wraps the replay memory instance to allow a modular transformation.

        This class is the base class for all wrappers. The subclass could override
        some methods to change the behavior of the original environment without touching the
        original code.

        .. note::
            Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """

    def __init__(self, replay_buffer: ReplayMemory):
        self.replay_buffer = replay_buffer

    def add(self, experience_dict):
        self.replay_buffer.add(experience_dict)

    def sample(self):
        return self.replay_buffer.sample()

    def temporal_sample(self):
        return self.replay_buffer.temporal_sample()

    def __getattr__(self, item):
        if "replay_buffer" in self.__dict__:
            return getattr(self.replay_buffer, item)
        else:
            raise AttributeError

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, item):
        return self.replay_buffer[item]
