"""
Yoinked from DLR-RM Stable Baselines 3:
https://github.com/DLR-RM/stable-baselines3/blob/23afedb254d06cae97064ca2aaba94b811d5c793/stable_baselines3/common/bit_flipping_env.py
Author: Antonin Raffin
License: MIT License  (https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE)
"""

from collections import OrderedDict
import typing as T

import numpy as np
import gym

spaces = gym.spaces
from franQ.Env import wrappers
from franQ.Env.conf import EnvConf


class CartPoleGoalEnv(wrappers.Wrapper):
    """Wrapper around araffin's bitflipping env to make it work with our API"""

    def __init__(self, conf: EnvConf):
        max_steps = {
            "v0": 200,
            "v1": 500,
            "v2": 1000,
        }
        v = str(conf.name).split(sep="-v")[1]
        random = "random" in str(conf.name).split(sep="-v")[0].lower()
        if random:
            raise NotImplementedError("TODO :) Reward function needs to be rethought")
        env = _CartPoleGoal(
            max_steps=max_steps[f"v{v}"],
            randomize_target=random,
            match_position=False,
            match_angle=True,
        )
        env = wrappers.ObsDictRenameKey(env, old_name="observation", new_name="obs_1d")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            env = wrappers.NormalizeActions(env)
        super().__init__(env)
        # conf.compute_reward = env.compute_reward
    def get_reward_functor(self):
        return self.env.compute_reward



class _CartPoleGoal(wrappers.Wrapper):
    """
    CartPole but a GoalEnv which uses the desired setpoint state vector as a goal

    The goal is to match that given state vector, not necessarily to stay upright.

    In the continuous variant, if the ith action component has a value > 0,
    then the ith bit will be flipped.

    :param n_bits: (int) Number of bits to flip
    :param continuous: (bool) Whether to use the continuous actions version or not,
        by default, it uses the discrete one
    :param max_steps: (Optional[int]) Max number of steps, by default, equal to n_bits
    :param discrete_obs_space: (bool) Whether to use the discrete observation
        version or not, by default, it uses the MultiBinary one
    """

    def __init__(self, max_steps: T.Optional[int] = None, randomize_target=False,
                 match_position=False, match_angle=True):
        from gym.envs.classic_control.cartpole import CartPoleEnv
        env: CartPoleEnv = CartPoleEnv()
        super().__init__(env)
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        # import math
        # env.theta_threshold_radians = 90 * 2 * math.pi / 360
        goal_high = np.array([
            env.observation_space.high[0],
            env.observation_space.high[2],
        ], dtype=np.float32)
        goal_low = np.array([
            env.observation_space.low[0],
            env.observation_space.low[2],
        ], dtype=np.float32)
        goal_space = spaces.Box(goal_low, goal_high, dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation": env.observation_space,
                "achieved_goal": goal_space,
                "desired_goal": goal_space,
            }
        )

        self.desired_goal = np.array([0, 0])
        self.goal_mask = np.array([int(match_position), int(match_angle)])

        self.randomize_target = randomize_target
        if max_steps in (None, 0):
            max_steps = np.inf
        self._max_episode_steps = max_steps
        self.reset()

    def _get_obs(self) -> OrderedDict:
        """
        Helper to create the observation.

        :return: (OrderedDict<int or ndarray>)
        """
        return OrderedDict(
            [
                ("observation", self.obs.copy()),
                ("achieved_goal",
                 np.array((self.obs[0], self.obs[2])).reshape(self.observation_space.spaces["achieved_goal"].shape)),
                ("desired_goal", self.desired_goal.copy()),
            ]
        )

    def reset(self) -> OrderedDict:
        self.current_step = 0
        if self.randomize_target:
            self.desired_goal = (self.observation_space.spaces["desired_goal"].sample() * self.goal_mask) / 3
        else:
            self.desired_goal = np.array([0, 0])
        self.obs = self.env.reset()

        return self._get_obs()

    def step(self, action: T.Union[np.ndarray, int]):
        self.obs, _, fail, info = self.env.step(action)

        info["fail"] = fail

        obs = self._get_obs()
        reward, done = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        done = done or self.current_step >= self._max_episode_steps
        return obs, reward, done, info

    @staticmethod
    def compute_reward(achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> T.Tuple[float, bool]:
        # Deceptive reward: it is positive only when the goal is achieved
        if info.get("fail",False):
            return -1, True
        if np.allclose(achieved_goal, desired_goal):
            return 1.0, False
        return 0.0, False


if __name__ == '__main__':

    def main():
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        env = _CartPoleGoal(randomize_target=False, match_position=False)

        P, I, D = 0.1, 0.01, 0.5

        for i_episode in range(20):
            obs = env.reset()
            desired_state = obs["desired_goal"]

            state = obs["achieved_goal"]
            integral = 0
            prev_error = 0
            for t in range(500):
                env.render()
                error = state - desired_state

                integral += error
                derivative = error - prev_error
                prev_error = error

                pid = np.dot(P * error + I * integral + D * derivative, env.goal_mask)
                action = sigmoid(pid)
                action = np.round(action).astype(np.int32)

                obs, reward, done, info = env.step(action)
                state = obs["achieved_goal"]
                desired_state = obs["desired_goal"]

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()


    main()
