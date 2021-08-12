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


class ClassicGoalEnv(wrappers.Wrapper):
    def __init__(self, conf: EnvConf):
        tasks = {
            "CartPole-v1": CartPoleGoalEnv,
            "Acrobot-v1": AcrobotGoalEnv,
        }
        env = tasks[conf.name]()
        env = wrappers.ObsDictRenameKey(env, old_name="observation", new_name="obs_1d")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            env = wrappers.NormalizeActions(env)
        super().__init__(env)

    def get_reward_functor(self):
        return self.env.compute_reward


class CartPoleGoalEnv(wrappers.Wrapper):
    """
    CartPole but a GoalEnv which uses the desired setpoint state vector as a goal
    The goal is to match that given state vector, not necessarily to stay upright.
    """

    def __init__(self, ):
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
        max_steps: T.Optional[int] = 500
        randomize_target = False
        match_position = False
        match_angle = True
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
        if self.current_step >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> T.Tuple[float, bool]:
        # Deceptive reward: it is positive only when the goal is achieved
        if info.get("fail", False):
            return -1, True
        if np.allclose(achieved_goal, desired_goal):
            return 1.0, False
        return 0.0, False


class AcrobotGoalEnv(wrappers.Wrapper):
    """
    Acrobot but a GoalEnv which uses the desired setpoint state vector as a goal
    The goal is to match that given state vector, not necessarily to stay upright.
    """

    def __init__(self, ):
        from gym.envs.classic_control.acrobot import AcrobotEnv
        env: AcrobotEnv = AcrobotEnv()
        super().__init__(env)
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        # import math
        # env.theta_threshold_radians = 90 * 2 * math.pi / 360
        max_steps: T.Optional[int] = 500
        goal_high = np.array([2.0], dtype=np.float32)
        goal_space = spaces.Box(-goal_high, goal_high, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": env.observation_space,
            "achieved_goal": goal_space,
            "desired_goal": goal_space,
        })

        self.desired_goal = np.zeros_like(goal_high) + 1.0
        self._max_episode_steps = float("inf") if max_steps in (None, 0) else max_steps
        self.reset()

    def _get_obs(self) -> OrderedDict:
        """
        Helper to create the observation.

        :return: (OrderedDict<int or ndarray>)
        """
        s = self.obs.copy()
        achieved_goal = np.zeros_like(self.desired_goal) - np.cos(s[0]) - np.cos(s[1] + s[0])
        return OrderedDict([
            ("observation", s),
            ("achieved_goal", achieved_goal),
            ("desired_goal", self.desired_goal.copy()),
        ])

    def reset(self) -> OrderedDict:
        self.current_step = 0
        self.obs = self.env.reset()
        return self._get_obs()

    def step(self, action: T.Union[np.ndarray, int]):
        self.obs, _, _, info = self.env.step(action)
        obs = self._get_obs()
        reward, done = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        if self.current_step >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> T.Tuple[float, bool]:
        done = achieved_goal >= desired_goal
        reward = 1.0 if done else 0.0
        return reward, done


if __name__ == '__main__':

    def main():
        env = AcrobotGoalEnv()
        for i_episode in range(5):
            obs = env.reset()
            min_ag = float("inf")
            max_ag = -float("inf")
            for t in range(int(1e6)):
                env.render()
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                ag = obs['achieved_goal']
                min_ag = min(ag, min_ag)
                max_ag = max(ag, max_ag)
                from franQ.common_utils import numpy_set_print_decimal_places
                numpy_set_print_decimal_places(2)
                print(f"{ag}\t{min_ag}\t{max_ag}\t{obs['observation']}")
                import time
                time.sleep(0.05)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()


    main()

if __name__ == '__main__':

    def main():
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        env = CartPoleGoalEnv()

        P, I, D = 0.1, 0.01, 0.5

        for i_episode in range(5):
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
