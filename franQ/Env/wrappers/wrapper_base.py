"""
Defining my own custom wrapper base class because the one in gym has undesired behavior around:
    - __getattr__() method: this should get private members too eg _max_episode_steps for time limit
    - __str__ and __name__ : creates a crash when a debugger steps into a wrapper before it has assigned the env member
"""

class Wrapper():
    r"""Wraps the environment to allow a modular transformation.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    .. note::

        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.

    """

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

    def __getattr__(self, name):
        if "env" in self.__dict__:
            print(name)
            return getattr(self.env, name)

    @property
    def spec(self):
        if "env" in self.__dict__:
            return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    def __str__(self):
        if "env" in self.__dict__:
            return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    @property
    def unwrapped(self):
        if "env" in self.__dict__:
            return self.env.unwrapped


class RewardWrapper(Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        raise NotImplementedError

class ActionWrapper(Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        raise NotImplementedError

    def reverse_action(self, action):
        raise NotImplementedError

class ObservationWrapper(Wrapper):
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        raise NotImplementedError


class AbstractAgent:
    def act(self):
        ...