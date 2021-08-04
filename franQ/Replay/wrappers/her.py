import numpy as np
import typing as T
from collections import deque, defaultdict
from .wrapper_base_class import ReplayMemoryWrapper
from .nstep_return import calculate_montecarlo_return


class HindsightNStepReplay(ReplayMemoryWrapper):
    """Hindsight Experience Replay with MC Lowerbounds
    This hinges on the assumptions:
    - Episode terminates when goal is reached!

    NOTE: NOT COMPATIBLE WITH SQUASH_REWARDS OR NSTEP RETURNS WRAPPERS!
    """

    def __init__(self, replay_buffer, compute_reward: T.Callable,ignore_keys=('info',)):

        ReplayMemoryWrapper.__init__(self, replay_buffer)
        self.compute_reward = compute_reward
        self._reset()
        self._ignored_keys = ignore_keys


    def _reset(self):
        self.buffers = defaultdict(deque)

    def add(self, experience: T.Dict[str, np.ndarray]):
        """Holds experiences and calculates n-step return before dumping to replay"""

        # Note: deques are filled FIFO from left[0], so oldest entry is at [-1]
        for key, value in experience.items():
            self.buffers[key].appendleft(value)

        if experience["episode_done"]:
            self._flush()
            self._hindsight_flush()
            self._reset()  # clear all buffers

    def _flush(self):
        """
        Forward the results stored in the buffer
        :return:
        """
        for i in reversed(range(len(self.buffers["reward"]))):
            new_xp_tuple = {k: v[i] for k, v in self.buffers.items()
                            if (len(v) and (k not in self._ignored_keys))}
            # Note: we are skipping any entries that aren't put in the buffer (eg for "next_", it is not
            # necessary to fill even though it is in the memories dict
            self.replay_buffer.add(new_xp_tuple)

    def _hindsight_flush(self):
        # Calculate what our rewards WOULD HAVE BEEN if we were working towards the state we ended up at
        hindsight_goal = self.buffers["achieved_goal"][0]
        reward = []
        done = []
        step = []
        for i, (ag, info) in enumerate(zip(self.buffers["achieved_goal"], self.buffers["info"])):
            goal_reward, d = self.compute_reward(ag, hindsight_goal, info)
            # Calculate the component of the reward that is NOT due to the desired_goal at the time
            # (ie things that just happen in the environment)
            goal_agnostic_reward = (
                    self.buffers["reward"][i] -
                    self.compute_reward(ag, self.buffers["desired_goal"][i], info)[0]
            )
            r = goal_agnostic_reward + goal_reward

            # Splice them into synthetic episodes and calculate retroactive flags and stats (eg MC) based on this split
            if d or (i==0): # New synthetic episode
                reward.append([])
                step.append([])
            # put rewards into last synthetic episode
            reward[-1].append(r)
            step[-1].append(self.buffers["episode_step"][i])
            done.append(d) # this doesn't need to be folded into episode lists because we'll just unfold
            # later when pushing to the true replay

        # Calculate statistics and unfold episodes (concatenate)
        reward = np.concatenate([np.asarray(r) for r in reward], axis=-1)
        step = np.concatenate([np.asarray(s) - s[-1] for s in step], axis=-1)

        # Push to experience replay
        for i in reversed(range(len(reward))):
            new_xp_tuple = {k: v[i] for k, v in self.buffers.items()
                            if (len(v) and (k not in self._ignored_keys))}
            # Note: we are skipping any entries that aren't put in the buffer (eg for "next_", it is not
            # necessary to fill even though it is in the memories dict
            new_xp_tuple["desired_goal"] = hindsight_goal
            new_xp_tuple["task_done"] = done[i]
            new_xp_tuple["episode_step"] = step[i]
            new_xp_tuple["reward"] = reward[i]
            self.replay_buffer.add(new_xp_tuple)
