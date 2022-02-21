import numpy as np
import numba
from typing import Dict, Iterable
from collections import deque, defaultdict
from .wrapper_base_class import ReplayMemoryWrapper


class NStepReturn(ReplayMemoryWrapper):
    """Calculate the MC return before storing it in the replay memory."""

    def __init__(self, replay_buffer, n_step, discount,
                 reward_name="reward", return_name="mc_return", done_name="episode_done"):
        if n_step > 1e4:
            print("Note: NStepReturn stores ALL experiences in RAM while calculating return. \n"
                    "Time and memory costs are O(n) in n_step, so careful how large you srt this thing")
        ReplayMemoryWrapper.__init__(self, replay_buffer)
        self.n_step, self.discount, self.reward_name, self.return_name, self.done_name = n_step, discount, reward_name, return_name, done_name
        self._reset()

    def _reset(self):
        self.buffers = defaultdict(deque)

    def add(self, experience: Dict[str, np.ndarray]):
        """Holds experiences and calculates n-step return before dumping to replay"""

        # Note: deques are filled FIFO from left[0], so oldest entry is at [-1]
        for key, value in experience.items():
            self.buffers[key].appendleft(value)
        assert self.reward_name in experience

        if experience[self.done_name]:
            self._flush()
        elif len(self.buffers[self.reward_name]) == self.n_step:
            self._pop()

    def _flush(self):
        # Purges all in-memory buffers at the end of the episode
        reward_buffer = self.buffers[self.reward_name]
        mc_return = calculate_montecarlo_return(reward_buffer, self.discount)

        # makes montecarlo return a new experience tuple entry
        for i in reversed(range(len(mc_return))):
            new_xp_tuple = {k: v[i] for k, v in self.buffers.items() if len(v)}
            # Note: we are skipping any entries that aren't put in the buffer (eg for "next_", it is not
            # necessary to fill even though it is in the memories dict
            new_xp_tuple[self.return_name] = mc_return[i]
            self.replay_buffer.add(new_xp_tuple)
        self._reset()  # clear all buffers

    def _pop(self):
        # For use when buffer is full. Removes the oldest entry
        reward_buffer = self.buffers[self.reward_name]
        mc_return = calculate_montecarlo_return(reward_buffer, self.discount)

        new_xp_tuple = {k: v[-1] for k, v in self.buffers.items()}  # by convention, oldest is [-1]
        new_xp_tuple[self.return_name] = mc_return[-1]
        self.replay_buffer.add(new_xp_tuple)


def calculate_montecarlo_return(rewards: Iterable, gamma):
    rewards = np.asarray(rewards, dtype=np.float32).squeeze()
    shape = rewards.shape
    if len(shape) == 0:
        return np.reshape(rewards,(1,))
    _inner(rewards, gamma, shape[0])
    return rewards


@numba.njit
def _inner(rewards, gamma, size):
    for i in range(1, size):
        rewards[i] += rewards[i - 1] * gamma
