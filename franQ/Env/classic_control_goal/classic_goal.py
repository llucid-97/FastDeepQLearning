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
            "MountainCar-v0": MountainCarGoalEnv,
            "Pendulum-v0": PendulumGoalEnv,
            "PendulumSparse-v0": PendulumSparseGoalEnv,
        }
        env = tasks[conf.name]()
        env = wrappers.ObsDictRenameKey(env, old_name="observation", new_name="obs_1d")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            env = wrappers.NormalizeActions(env)
        super().__init__(env)

    def get_reward_functor(self):
        return self.env.compute_reward


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
        s = self.env.state
        achieved_goal = np.zeros_like(self.desired_goal) - np.cos(s[0]) - np.cos(s[1] + s[0])
        return OrderedDict([
            ("observation", self.obs.copy()),
            ("achieved_goal", achieved_goal),
            ("desired_goal", self.desired_goal.copy()),
        ])

    def reset(self) -> OrderedDict:
        self.current_step = 0
        self.obs = self.env.reset()
        return self._get_obs()

    def step(self, action: T.Union[np.ndarray, int]):
        self.obs, _, terminal, info = self.env.step(action)
        obs = self._get_obs()
        reward, done = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        if self.current_step >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        # Deceptive reward: it is positive only when the goal is achieved
        reward = 0.0 if (achieved_goal >= desired_goal).all() else -1.0
        done = reward == 0
        return reward, done


class PendulumGoalEnv(wrappers.Wrapper):
    """
    Pendulum but a GoalEnv which uses the desired setpoint state vector as a goal
    The goal is to match that given state vector, not necessarily to stay upright.
    """

    def __init__(self, ):
        from gym.envs.classic_control.pendulum import PendulumEnv
        env: PendulumEnv = PendulumEnv()
        super().__init__(env)
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        # import math
        # env.theta_threshold_radians = 90 * 2 * math.pi / 360
        max_steps: T.Optional[int] = 500
        goal_high = np.array([np.pi, env.max_speed], dtype=np.float32)
        goal_space = spaces.Box(-goal_high, goal_high, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": env.observation_space,
            "achieved_goal": goal_space,
            "desired_goal": goal_space,
        })

        self.desired_goal = np.zeros_like(goal_high)
        self._max_episode_steps = float("inf") if max_steps in (None, 0) else max_steps
        self.reset()

    def _get_obs(self) -> OrderedDict:
        """
        Helper to create the observation.

        :return: (OrderedDict<int or ndarray>)
        """

        return OrderedDict([
            ("observation", self.obs.copy()),
            ("achieved_goal", np.asarray(self.state, dtype=np.float32)),
            ("desired_goal", self.desired_goal.copy()),
        ])

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        # Deceptive reward: it is positive only when the goal is achieved

        th, thdot = tuple(achieved_goal)  # th := theta
        th2, thdot2 = tuple(desired_goal)  # th := theta
        u = info.get("u", [0.0])
        u = np.clip(u, -self.env.max_torque, self.env.max_torque)[0]
        costs = (
                abs(th - th2)  # ** 2  # Penalize angle difference from target
                + .1 * abs(thdot)  # ** 2  # Penalize velocity
                + .001 * (u ** 2)  # Penalize torque
        )
        # reward = 0.0 if (achieved_goal >= desired_goal).all() else -1.0
        return -costs, False

    def reset(self) -> OrderedDict:
        self.current_step = 0
        self.obs = self.env.reset()
        self.state = self.env.state.copy()
        return self._get_obs()

    def step(self, action: T.Union[np.ndarray, int]):
        from gym.envs.classic_control.pendulum import angle_normalize

        self.state = self.env.state.copy()
        self.state[0] = angle_normalize(self.state[0])
        self.obs, _, terminal, info = self.env.step(action)
        info["u"] = action
        obs = self._get_obs()
        reward, done = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        if self.current_step >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info


class PendulumSparseGoalEnv(PendulumGoalEnv):
    """PundulumGoalEnv except with a sparse reward of +1 on reaching the goal state and -1 everywhere else"""

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        th, thdot = tuple(achieved_goal)
        th2, thdot2 = tuple(achieved_goal)

        u = info["u"]
        u = np.clip(u, -self.env.max_torque, self.env.max_torque)[0]
        rewards = (
                np.pi * float(np.allclose(th, th2, atol=1e-1)) - 1.0  # Encourage reaching target angle
                - .1 * thdot ** 2  # Penalize velocity
                - .001 * (u ** 2)  # Penalize torque
        )
        return rewards, False


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
            return -1.0, True
        if np.allclose(achieved_goal, desired_goal):
            return 1.0, False
        return 0.0, False


class MountainCarGoalEnv(wrappers.Wrapper):
    def __init__(self, ):
        from gym.envs.classic_control.mountain_car import MountainCarEnv
        import copy
        env: MountainCarEnv = MountainCarEnv()
        super().__init__(env)
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        # import math
        # env.theta_threshold_radians = 90 * 2 * math.pi / 360
        max_steps: T.Optional[int] = 500
        self.observation_space = spaces.Dict({
            "observation": env.observation_space,
            "achieved_goal": env.observation_space,
            "desired_goal": env.observation_space,
        })

        self.desired_goal = self.observation_space["desired_goal"].sample()
        self.desired_goal[0] = env.goal_position
        self.desired_goal[1] = env.goal_velocity
        self._max_episode_steps = float("inf") if max_steps in (None, 0) else max_steps
        self.reset()

    def _get_obs(self) -> OrderedDict:
        """
        Helper to create the observation.

        :return: (OrderedDict<int or ndarray>)
        """

        return OrderedDict([
            ("observation", self.obs.copy()),
            ("achieved_goal", self.obs.copy()),
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
        done = np.all(achieved_goal >= desired_goal)
        reward = 50.0 if done else -1.0
        return reward, done


if __name__ == '__main__':

    def main():
        for Env_T in (
                # MountainCarGoalEnv,
                # CartPoleGoalEnv,
                # AcrobotGoalEnv,
                PendulumGoalEnv,
        ):
            env = Env_T()

            for i_episode in range(20000):
                obs = env.reset()
                for t in range(int(1e6)):
                    env.render()
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    assert isinstance(obs, dict)
                    assert "observation" in obs
                    assert "achieved_goal" in obs
                    assert "desired_goal" in obs
                    for k in obs:
                        if isinstance(env.observation_space.spaces[k], gym.spaces.Box):
                            assert obs[k].shape == env.observation_space.spaces[k].shape
                            assert np.all(obs[k] <= env.observation_space.spaces[k].high)
                            assert np.all(obs[k] >= env.observation_space.spaces[k].low)
                    if done:
                        print("Episode finished after {} timesteps".format(t + 1))
                        break
            env.close()


    main()
