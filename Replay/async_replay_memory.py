from Replay.replay_memory import ReplayMemory, OversampleError
from torch import multiprocessing as mp
from threading import Thread
import time


class AsyncReplayMemory:
    """Creates a replay memory in another process and sets up an API to access it"""

    def __init__(self, maxlen, batch_size, temporal_len):
        self.batch_size = batch_size
        self._temporal_len = temporal_len
        self._q_sample = mp.Queue(maxsize=3)
        self._q_sample_temporal = mp.Queue(maxsize=3)
        self._q_add = mp.Queue(maxsize=3)
        self._len = mp.Value("i", 0)
        self._maxlen = maxlen
        mp.Process(
            target=_child_process,
            args=[maxlen, batch_size, temporal_len, self._q_sample, self._q_add, self._q_sample_temporal]
        ).start()

    def add(self, experience_dict):
        self._len.value = min((self._len.value + 1), self._maxlen)
        self._q_add.put(experience_dict)

    def sample(self):
        return self._q_sample.get()

    def temporal_sample(self):  # sample [Batch, Time, Experience]
        return self._q_sample_temporal.get()

    def __len__(self):
        return self._len.value


def _child_process(maxlen, batch_size, temporal_len, sample_q: mp.Queue, add_q: mp.Queue, temporal_q: mp.Queue):
    """Creates replay memory instance and parallel threads to add and sample memories"""
    replay_T = ReplayMemory  # ReplayMemory
    replay = replay_T(maxlen, batch_size, temporal_len)

    def sample():
        while True:
            try:
                sample_q.put(replay.sample())
            except OversampleError:
                time.sleep(1)

    def sample_temporal():
        while True:
            try:
                temporal_q.put(replay.temporal_sample())
            except OversampleError:
                time.sleep(1)

    def add():
        while True:
            replay.add(add_q.get())

    threads = [Thread(target=sample), Thread(target=add), Thread(target=sample_temporal)]
    [t.start() for t in threads]
    [t.join() for t in threads]
