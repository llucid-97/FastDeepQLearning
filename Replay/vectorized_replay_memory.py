import typing as T
import numpy as np
from .replay_memory import OversampleError

class VectorizedReplayMemory:
    """
    A variant of the replay memory which stores all experiences as a vector in memory rather than a deque.
    This allows for more efficient batched accesses.
    It relies on the assumptions that:
        - The keys, dtype and shape of all entries added to it remain the same as the first
        - Each environment has its unique instance so memories are added chronologically
    """

    def __init__(self, maxlen, batch_size, temporal_len):
        self.batch_size, self.temporal_len, self.maxlen = batch_size, temporal_len, maxlen
        self.top, self.curr_len = 0, 0
        self.memory: T.Dict[str, np.ndarray] = {}

    def jit_initialize(self, experience_dict: dict):
        """This method is called the first time an experience is added.
        It initializes the memory data structure to have the right shape and dtype to store all subsequent entries"""
        for k, v in experience_dict.items():
            if isinstance(v, np.ndarray):
                shape = (self.maxlen,) + tuple(v.shape)
                dtype = v.dtype
            else:
                assert np.isclose(np.float32(v),
                                  v), "Anything thats not a numpy array must be representable as a float32 for numeric stability"
                shape = (self.maxlen, 1)
                dtype = np.float32
            self.memory[k] = np.zeros(shape, dtype)

    def add(self, experience_dict: dict):
        if len(self.memory) == 0:
            self.jit_initialize(experience_dict)

        for k, v in experience_dict.items():
            self.memory[k][self.top] = v

        self.top = (self.top + 1) % self.maxlen
        self.curr_len = max(self.top, self.curr_len)

    def sample(self) -> T.Dict[str, np.ndarray]:
        # sample [Batch, Experience]
        if len(self) < self.batch_size: raise OversampleError("Trying to sample more memories than available!")
        idxes = np.random.randint(0, self.curr_len, self.batch_size)
        return self[idxes]

    def temporal_sample(self):
        # sample [Temporal, Batch, Experience]
        if len(self) < self.temporal_len: raise OversampleError("Trying to sample more memories than available!")
        if len(self) < self.batch_size: raise OversampleError("Trying to sample more memories than available!")
        curr_len = self.curr_len
        batch = np.random.randint(0, curr_len, self.batch_size)
        temporal = np.arange(self.temporal_len)
        idxes = np.reshape(temporal, (-1, 1)) + np.reshape(batch, (1, -1))
        idxes = idxes % curr_len
        return self[idxes]

    def __getitem__(self, idxes) -> T.Dict[str, np.ndarray]:
        idxes = np.asarray(idxes)
        return {k: v[idxes] for k, v in self.memory.items()}

    def __len__(self):
        return self.curr_len