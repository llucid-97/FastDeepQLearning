import numpy as np
import jax.numpy as jnp
import typing as T, random
from collections import deque, defaultdict
from .wrapper_base_class import ReplayMemoryWrapper
import numba
import jax


class HindsightVmapWrite(ReplayMemoryWrapper):
    """Hindsight Experience Replay with MC Lowerbounds
    This hinges on the assumptions:
    - Episode terminates when goal is reached!

    This variant computes `num_virtual_goals` virtual goals instead of 1, and samples from them at train time
    """

    def __init__(self, replay_buffer, compute_reward: T.Callable, ignore_keys=('info',), num_virtual_goals=32):
        super().__init__(replay_buffer)
        self.compute_reward = compute_reward
        self._reset()
        self._ignored_keys = ignore_keys
        self.num_virtual_goals = num_virtual_goals
        self._setup_vmap_functions()

    def _setup_vmap_functions(self):
        batch_compute_reward = jax.vmap(self.compute_reward)
        batch_compute_reward_broadcast = jax.vmap(self.compute_reward, in_axes=(0, None))

        def _virtual_episode_calc(virtual_goal, desired_goal, achieved_goal, reward, done, ):
            desired_reward, desired_done = batch_compute_reward(achieved_goal, desired_goal)
            virtual_reward, virtual_done = batch_compute_reward_broadcast(achieved_goal, virtual_goal)
            goal_agnostic_reward = reward - desired_reward
            # print(f"shapes | dgr={np.shape(desired_reward)} dgd={np.shape(desired_done)}")
            # print(f"shapes | vgr={np.shape(virtual_reward)} vgd={np.shape(virtual_done)}")
            # print(f"shapes | d={np.shape(done)} vgd={np.shape(virtual_done)}")

            goal_agnostic_done = jnp.logical_and(done, jnp.logical_not(desired_done))

            virtual_reward = goal_agnostic_reward + virtual_reward
            virtual_done = jnp.logical_or(goal_agnostic_done, virtual_done)

            return virtual_reward, virtual_done

        self.batch_compute_virtual_episodes = jax.vmap(_virtual_episode_calc, in_axes=(0, None, None, None, None))

    def _reset(self):
        self.buffers = defaultdict(deque)

    def add(self, experience: T.Dict[str, np.ndarray]):
        """Holds experiences and calculates n-step return before dumping to replay"""

        # Note: deques are filled FIFO from left[0], so oldest entry is at [-1]
        for key, value in experience.items():
            self.buffers[key].appendleft(value)

        if experience["episode_done"]:
            self._hindsight_flush()
            self._reset()  # clear all buffers

    def _select_virtual_goal(self):
        if self._mode == "random":
            import random
            return random.choice(self.buffers["achieved_goal"])

    def _hindsight_flush(self):
        # Calculate what our rewards WOULD HAVE BEEN if we were working towards the state we ended up at
        cpu = jax.devices("cpu")[0]
        to_jax = lambda x: jax.device_put(np.asarray(x), cpu)
        achieved_goal = to_jax(self.buffers["achieved_goal"])
        desired_goal = to_jax(self.buffers["desired_goal"])
        reward = to_jax(self.buffers["reward"])
        done = to_jax(self.buffers["task_done"])

        virtual_goals = achieved_goal[np.random.randint(0, achieved_goal.shape[0], size=self.num_virtual_goals)]
        virtual_rewards, virtual_dones = self.batch_compute_virtual_episodes(virtual_goals, desired_goal,
                                                                                       achieved_goal, reward, done)

        # Push to experience replay
        for i in reversed(range(len(reward))):
            new_xp_tuple = {k: v[i] for k, v in self.buffers.items()
                            if (len(v) and (k not in self._ignored_keys))}
            # Note: we are skipping any entries that aren't put in the buffer (eg for "next_", it is not
            # necessary to fill even though it is in the memories dict
            new_xp_tuple["virtual_goals"] = np.concatenate([virtual_goals[:, ], [new_xp_tuple["desired_goal"]]])
            new_xp_tuple["virtual_rewards"] = np.concatenate([virtual_rewards[:, i], [new_xp_tuple["reward"]]])
            new_xp_tuple["virtual_dones"] = np.concatenate([virtual_dones[:, i], [new_xp_tuple["task_done"]]])
            self.replay_buffer.add(new_xp_tuple)

        # TODO: Add task done and episode done as a filter confition to episode step


class HindsightVmapRead(ReplayMemoryWrapper):
    """Hindsight Experience Replay with MC Lowerbounds
    This hinges on the assumptions:
    - Episode terminates when goal is reached!

    This variant is the reader head. It is kept separate from the writer so we don't have to push the compute_function
    object across process boundaries. It can be stacked with it if you want to use both.

    This samples one of the virtual goals and replaces the real goal with it
    """

    def temporal_sample(self):
        xp_dict = super(HindsightVmapRead, self).temporal_sample()
        # choose the goal
        num_virtual_goals = xp_dict["virtual_goals"].shape[2]
        idx = random.randint(0, num_virtual_goals)

        xp_dict["desired_goal"] = xp_dict["virtual_goals"][:, :, idx]
        xp_dict["reward"] = xp_dict["virtual_rewards"][:, :, idx,None]
        xp_dict["task_done"] = xp_dict["virtual_dones"][:, :, idx,None]
        if "virtual_mc_return" in xp_dict:
            xp_dict["mc_return"] = xp_dict["virtual_mc_return"][:, :, idx,None]
        return self.cleanup(xp_dict)

    def cleanup(self, xp_dict):
        del xp_dict['virtual_goals']
        del xp_dict['virtual_rewards']
        del xp_dict['virtual_dones']
        if "virtual_mc_return" in xp_dict:
            del xp_dict["virtual_mc_return"]
        return xp_dict
