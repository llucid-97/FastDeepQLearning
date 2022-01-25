import typing as T
import numpy as np


class OversampleError(Exception): ...


class ReplayMemory:
    """
    A variant of the replay memory which stores all experiences as a vector in memory rather than a deque.
    This allows for more efficient batched accesses.
    It relies on the assumptions that:
        - The keys, dtype and shape of all entries added to it remain the same as the first
        - Each environment has its unique instance so memories are added chronologically
    """

    def __init__(self, maxlen, batch_size, temporal_len,**kwargs):
        self._batch_size, self._temporal_len, self._maxlen = batch_size, temporal_len, maxlen
        self._top, self._curr_len = 0, 0
        self.memory: T.Dict[str, np.ndarray] = {}

    def _jit_initialize(self, experience_dict: dict):
        """This method is called the first time an experience is added.
        It initializes the memory data structure to have the right shape and dtype to store all subsequent entries"""
        for k, v in experience_dict.items():
            if isinstance(v, np.ndarray):
                shape = (self._maxlen,) + tuple(v.shape)
                dtype = v.dtype
            else:
                assert np.isclose(np.float32(v),
                                  v), "Anything thats not a numpy array must be representable as a float32 for numeric stability"
                shape = (self._maxlen, 1)
                dtype = np.float32
            self.memory[k] = np.zeros(shape, dtype)

    def add(self, experience_dict: dict):
        if len(self.memory) == 0:
            self._jit_initialize(experience_dict)

        for k, v in experience_dict.items():
            self.memory[k][self._top] = v

        self._top = (self._top + 1) % self._maxlen
        self._curr_len = max(self._top, self._curr_len)

    def sample(self) -> T.Dict[str, np.ndarray]:
        # sample [Batch, Experience]
        if len(self) < self._batch_size: raise OversampleError("Trying to sample more memories than available!")
        idxes = np.random.randint(0, self._curr_len, self._batch_size)
        return self[idxes]

    def temporal_sample(self):
        # sample [Temporal, Batch, Experience]
        _len = len(self)
        if _len < (self._temporal_len * 2): raise OversampleError("Trying to sample more memories than available!")
        if _len < self._batch_size: raise OversampleError("Trying to sample more memories than available!")

        batch = np.random.randint(0, _len - self._temporal_len, self._batch_size)
        temporal = np.arange(self._temporal_len)
        idxes = np.reshape(temporal, (-1, 1)) + np.reshape(batch, (1, -1))
        idxes = idxes % _len
        return self[idxes]

    def __getitem__(self, idxes) -> T.Dict[str, np.ndarray]:
        idxes = np.asarray(idxes)
        return {k: v[idxes] for k, v in self.memory.items()}

    def __len__(self):
        return self._curr_len
