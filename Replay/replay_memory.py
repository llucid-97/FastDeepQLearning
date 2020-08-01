from collections import deque
import random
import typing as T


class OversampleError(Exception):
    pass


class ReplayMemory:
    """An in-memory FIFO queue which holds experiences from the environment.
    The agent samples from this as a dataset to train on.
    It relies on the assumption that  all data is kept in FIFO order, so each environment should have its own instance"""
    def __init__(self, maxlen, batch_size, temporal_len):
        self.memory = deque(maxlen=maxlen)
        self.batch_size, self.temporal_len = batch_size, temporal_len

    def add(self, experience_dict):
        self.memory.append(experience_dict)

    def sample(self) -> T.List[dict]:
        if len(self.memory) < self.batch_size: raise OversampleError("Trying to sample more memories than available!")
        return random.sample(self.memory, self.batch_size)  # sample without replacement

    def temporal_sample(self) -> T.List[T.List[dict]]:
        # sample [Batch, Time, Experience]
        if len(self) <= self.temporal_len: raise OversampleError("Trying to sample more memories than available!")
        if len(self) < self.batch_size: raise OversampleError("Trying to sample more memories than available!")
        ret = []
        for _ in range(self.batch_size):
            idx = random.randint(0, len(self) - self.temporal_len)
            ret.append([self.memory[idx + i] for i in range(self.temporal_len)])
        return ret

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]
