from collections import OrderedDict
import typing as T
import warnings
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
        reward, done = self.compute_reward(obs["achieved_goal"], obs["desired_goal"])
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        if self.current_step >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,):
        # Deceptive reward: it is positive only when the goal is achieved
        cond = (achieved_goal >= desired_goal).all()
        reward = 0.0 * cond  + (( -1.0) * (1-cond))
        done = reward == 0
        return reward, done


class PendulumGoalEnv(wrappers.Wrapper):
    """
    Pendulum but a GoalEnv which uses the desired setpoint state vector as a goal
    The goal is to match that given state vector, not necessarily to stay upright.
    """

    def __init__(self, ):
        raise NotImplementedError("This Env uses old style non-vectorizable compute reward functions."
                                  "\nThese are no longer supported. It will be updated in a later release")
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

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,):
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
        reward, done = self.compute_reward(obs["achieved_goal"], obs["desired_goal"])
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        if self.current_step >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info


class PendulumSparseGoalEnv(PendulumGoalEnv):
    """PundulumGoalEnv except with a sparse reward of +1 on reaching the goal state and -1 everywhere else"""

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        # Deceptive reward: it is positive only when the goal is achieved

        th, thdot = tuple(achieved_goal)  # th := theta
        th2, thdot2 = tuple(desired_goal)  # th := theta
        u = info.get("u", [0.0])
        u = np.clip(u, -self.env.max_torque, self.env.max_torque)[0]
        rewards = (
            (float(np.allclose(th, th2, atol=1e-1)) - 1.0)  # Encourage reaching target angle
        )
        return rewards, False


class CartPoleGoalEnv(wrappers.Wrapper):
    """
    CartPole but a GoalEnv which uses the desired setpoint state vector as a goal
    The goal is to match that given state vector, not necessarily to stay upright.
    """

    def __init__(self, ):
        raise NotImplementedError("This Env uses old style non-vectorizable compute reward functions."
                                  "\nThese are no longer supported. It will be updated in a later release")
        from gym.envs.classic_control.cartpole import CartPoleEnv
        env: CartPoleEnv = CartPoleEnv()
        super().__init__(env)
        goal_high = np.array([
            env.observation_space.high[0],
            env.observation_space.high[2],
        ], dtype=np.float32)
        goal_low = np.array([
            env.observation_space.low[0],
            env.observation_space.low[2],
        ], dtype=np.float32)
        goal_space = spaces.Box(goal_low, goal_high, dtype=np.float32)
        self.observation_space = spaces.Dict({"observation": env.observation_space,
                                              "achieved_goal": goal_space,
                                              "desired_goal": goal_space,
                                              })

        self.desired_goal = np.array([0, 0], dtype=np.float32)
        max_steps: T.Optional[int] = 500
        self._max_episode_steps = float("inf") if max_steps in (None, 0) else max_steps
        self.reset()

    def _get_obs(self) -> OrderedDict:
        return OrderedDict(
            [
                ("observation", self.obs.copy()),
                ("achieved_goal", np.array((self.obs[0], self.obs[2]), dtype=np.float32)),
                ("desired_goal", self.desired_goal.copy()),
            ]
        )

    def reset(self) -> OrderedDict:
        self.current_step = 0
        self.obs = self.env.reset()
        return self._get_obs()

    def step(self, action: T.Union[np.ndarray, int]):
        self.obs, _, fail, info = self.env.step(action)

        info["fail"] = fail

        obs = self._get_obs()
        reward, done = self.compute_reward(obs["achieved_goal"], obs["desired_goal"])
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        if self.current_step >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,) -> T.Tuple[float, bool]:
        if info.get("fail", False):
            return -1.0, True
        if np.allclose(achieved_goal[0], desired_goal[0], atol=1e-2):
            # Do not reward angle here because hindsight will proc and falsely incentivise it
            return 1.0, False
        return .1, False  # Infinite Run. Incentivize survival


class MountainCarGoalEnv(wrappers.Wrapper):
    def __init__(self, ):
        from gym.envs.classic_control.mountain_car import MountainCarEnv
        env: MountainCarEnv = MountainCarEnv()
        super().__init__(env)
        max_steps: T.Optional[int] = 500
        self.observation_space = spaces.Dict({
            "observation": env.observation_space,
            "achieved_goal": env.observation_space,
            "desired_goal": env.observation_space,
        })
        self.env: MountainCarEnv

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
        reward, done = self.compute_reward(obs["achieved_goal"], obs["desired_goal"])
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        if self.current_step >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> T.Tuple[float, bool]:
        position, velocity = achieved_goal
        goal_position, _ = desired_goal
        done = bool(position >= goal_position)  # and velocity >= self.env.goal_velocity)
        reward = float(done) - 1.0
        return reward, done


if __name__ == '__main__':

    def main():
        for Env_T in (
                MountainCarGoalEnv,
                CartPoleGoalEnv,
                AcrobotGoalEnv,
                PendulumGoalEnv,
                PendulumSparseGoalEnv,
        ):
            env = Env_T()

            for i_episode in range(2):
                _ = env.reset()
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
