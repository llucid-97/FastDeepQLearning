"""
Yoinked from DLR-RM Stable Baselines 3:
https://github.com/DLR-RM/stable-baselines3/blob/23afedb254d06cae97064ca2aaba94b811d5c793/stable_baselines3/common/bit_flipping_env.py
Author: Antonin Raffin
License: MIT License  (https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE)
"""

from collections import OrderedDict
from typing import Optional, Union, Callable

import numpy as np
import gym

spaces = gym.spaces
from .wrappers import wrapper_base, common
from .conf import EnvConf


class BitFlippingEnv(wrapper_base.Wrapper):
    """Wrapper around araffin's bitflipping env to make it work with ours"""

    def __init__(self, conf: EnvConf):
        env = _BitFlippingEnv(
            n_bits=int(str(conf.name).split(sep="-v")[1]),
            randomize_target=str(conf.name).split(sep="-v")[0] == "random",
            max_steps=int(str(conf.name).split(sep="-v")[1]) * 2,
        )
        env = common.ObsDictRenameKey(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            env = common.NormalizeActions(env)
        super().__init__(env)

    def get_reward_functor(self) -> Callable:
        return self.env.compute_reward


class _BitFlippingEnv(gym.GoalEnv):
    """
    Simple bit flipping env, useful to test HER.
    The goal is to flip all the bits to get a vector of ones.
    In the continuous variant, if the ith action component has a value > 0,
    then the ith bit will be flipped.

    :param n_bits: (int) Number of bits to flip
    :param continuous: (bool) Whether to use the continuous actions version or not,
        by default, it uses the discrete one
    :param max_steps: (Optional[int]) Max number of steps, by default, equal to n_bits
    :param discrete_obs_space: (bool) Whether to use the discrete observation
        version or not, by default, it uses the MultiBinary one
    """

    def __init__(self, n_bits: int = 10, continuous: bool = False, max_steps: Optional[int] = None,
                 discrete_obs_space: bool = False, randomize_target=False):
        super(_BitFlippingEnv, self).__init__()
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        if discrete_obs_space:
            # In the discrete case, the agent act on the binary
            # representation of the observation
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Discrete(2 ** n_bits - 1),
                    "achieved_goal": spaces.Discrete(2 ** n_bits - 1),
                    "desired_goal": spaces.Discrete(2 ** n_bits - 1),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.MultiBinary(n_bits),
                    "achieved_goal": spaces.MultiBinary(n_bits),
                    "desired_goal": spaces.MultiBinary(n_bits),
                }
            )

        self.obs_space = spaces.MultiBinary(n_bits)

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=(n_bits,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(n_bits)
        self.continuous = continuous
        self.discrete_obs_space = discrete_obs_space
        self.state = None
        self.desired_goal = np.ones((n_bits,))
        self.randomize_target = randomize_target
        self.n_bits = n_bits
        if max_steps is None:
            max_steps = n_bits
        self.max_steps = max_steps
        self.current_step = 0
        self.reset()

    def convert_if_needed(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """
        Convert to discrete space if needed.

        :param state: (np.ndarray)
        :return: (np.ndarray or int)
        """
        if self.discrete_obs_space:
            # The internal state is the binary representation of the
            # observed one
            return int(sum([state[i] * 2 ** i for i in range(len(state))]))
        return state

    def _get_obs(self) -> OrderedDict:
        """
        Helper to create the observation.

        :return: (OrderedDict<int or ndarray>)
        """
        return OrderedDict(
            [
                ("observation", self.convert_if_needed(self.state.copy())),
                ("achieved_goal", self.convert_if_needed(self.state.copy())),
                ("desired_goal", self.convert_if_needed(self.desired_goal.copy())),
            ]
        )

    def reset(self) -> OrderedDict:
        self.current_step = 0
        self.desired_goal = np.random.randint(0, 2, size=self.n_bits)
        self.state = self.obs_space.sample()
        return self._get_obs()

    def step(self, action: Union[np.ndarray, int]):
        if self.continuous:
            self.state[action > 0] = 1 - self.state[action > 0]
        else:
            self.state[action] = 1 - self.state[action]
        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None)
        done = reward == 0
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        info = {}
        if self.current_step >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        done = done or self.current_step >= self.max_steps
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        # Deceptive reward: it is positive only when the goal is achieved
        if self.discrete_obs_space:
            reward = 0.0 if achieved_goal == desired_goal else -1.0
        else:
            reward = 0.0 if (achieved_goal == desired_goal).all() else -1.0
        done = reward == 0
        return reward, done

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "rgb_array":
            return self.state.copy()
        print(self.state)

    def close(self) -> None:
        pass


if __name__ == '__main__':
    def main():
        env = _BitFlippingEnv(n_bits=4, max_steps=1000)
        while True:
            env.reset()
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, done, _ = env.step(action)
                print(f"obs={obs['observation']}\t reward={reward}")


    main()
