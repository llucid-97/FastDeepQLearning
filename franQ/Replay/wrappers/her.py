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

    def __init__(self, replay_buffer, n_step, discount, compute_reward: T.Callable):
        if n_step > 1e4:
            print("Note: NStepReturn stores ALL experiences in RAM while calculating return. \n"
                  "Time and memory costs are O(n) in n_step, so careful how large you srt this thing")
        ReplayMemoryWrapper.__init__(self, replay_buffer)
        self.n_step, self.discount = n_step, discount
        self.compute_reward = compute_reward
        self._reset()

    def _reset(self):
        self.buffers = defaultdict(deque)

    def add(self, experience: T.Dict[str, np.ndarray]):
        """Holds experiences and calculates n-step return before dumping to replay"""

        # Note: deques are filled FIFO from left[0], so oldest entry is at [-1]
        for key, value in experience.items():
            self.buffers[key].appendleft(value)

        if experience["episode_done"]:
            self._flush()
            task_successful = \
            self.compute_reward(self.buffers["achieved_goal"][0], self.buffers["desired_goal"][0], None)[1]
            if not task_successful:
                # generate a virtual episode using hindsight
                self._hindsight_flush()
            self._reset()  # clear all buffers

    def _flush(self):
        # Normal N-step Return calculation

        reward = np.asarray(self.buffers["reward"])
        mc_return = calculate_montecarlo_return(np.copy(reward), self.discount)
        for i in reversed(range(len(mc_return))):
            new_xp_tuple = {k: v[i] for k, v in self.buffers.items() if len(v)}
            # Note: we are skipping any entries that aren't put in the buffer (eg for "next_", it is not
            # necessary to fill even though it is in the memories dict
            new_xp_tuple["mc_return"] = mc_return[i]
            new_xp_tuple["reward"] = reward[i]
            self.replay_buffer.add(new_xp_tuple)

    def _hindsight_flush(self):
        hindsight_goal = self.buffers["achieved_goal"][0]
        # Calculate what our rewards WOULD HAVE BEEN if we were working towards the state we ended up at
        reward = []
        done = []
        step = []
        for i, ag in enumerate(self.buffers["achieved_goal"]):
            r, d = self.compute_reward(ag, hindsight_goal, None)
            # Calculate the component of the reward that is NOT due to our current desired_goal
            # (ie things that just happen in the environment)
            goal_agnostic_reward = self.buffers["reward"][i] - \
                                   self.compute_reward(ag, self.buffers["desired_goal"][0], None)[0]
            r = goal_agnostic_reward + r
            if d:
                reward.append([])
                step.append([])
            reward[-1].append(r)
            step[-1].append(self.buffers["episode_step"][i])
            done.append(d)
        mc_return = np.concatenate([calculate_montecarlo_return(r, self.discount) for r in reward], axis=-1)
        reward = np.concatenate([np.asarray(r) for r in reward], axis=-1)
        step = np.concatenate([np.asarray(s) - s[-1] for s in step], axis=-1)
        for i in reversed(range(len(mc_return))):
            new_xp_tuple = {k: v[i] for k, v in self.buffers.items() if len(v)}
            # Note: we are skipping any entries that aren't put in the buffer (eg for "next_", it is not
            # necessary to fill even though it is in the memories dict
            new_xp_tuple["desired_goal"] = hindsight_goal
            new_xp_tuple["task_done"] = done[i]
            new_xp_tuple["episode_step"] = step[i]
            new_xp_tuple["mc_return"] = mc_return[i]
            new_xp_tuple["reward"] = reward[i]
            self.replay_buffer.add(new_xp_tuple)

        # makes montecarlo return a new experience tuple entry

    def _pop(self):
        # This is broken in HER: the discontinuity would mean we won't be optimizing towards the right virtual_goal
        # TO work around this, we _flush, not really _pop,
        self.buffers["episode_step"][0] = -1  # signal that a discontinuity exists so it can be ignored during training
        self._flush()
