from collections import deque
import os

import cv2
from .wrapper_base import Wrapper, ObservationWrapper
import gym
import gym.spaces as spaces
import numpy as np

os.environ.setdefault("PATH", "")
cv2.ocl.setUseOpenCL(False)


class FrameStack(Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)

        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype,
        )

    # pylint: disable=method-hidden
    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    # pylint: disable=method-hidden
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=-1)


class ScaledFloatFrame(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        img = np.array(observation, dtype=np.float32)
        img /= 255.0
        return img


class NormalizeActions(Wrapper):
    """Makes env expect actions that are zero-mean and unit variance """

    def __init__(self, env):
        Wrapper.__init__(self, env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class FrameSkip(Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._skip = max(1, skip)

    # pylint: disable=method-hidden
    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None

        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    # pylint: disable=method-hidden
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)


class ObsDict(ObservationWrapper):
    """Transform the normal observation into an obs dict with a key given by arg"""

    def __init__(self, env, default_key="obs_1d"):
        super(ObsDict, self).__init__(env)
        self.observation_space = gym.spaces.Dict({default_key: self.env.observation_space})
        self._key = default_key

    def observation(self, observation):
        return {self._key: observation}

class ObsDictRenameKey(ObservationWrapper):
    """Renames a key for an obs dict"""

    def __init__(self, env, old_name="observation",new_name="obs_1d"):
        super(ObsDictRenameKey, self).__init__(env)
        old_obs_space = env.observation_space
        assert isinstance(old_obs_space,gym.spaces.Dict)
        import copy
        new_obs_space = copy.deepcopy(old_obs_space)
        new_obs_space.spaces[new_name] = new_obs_space.spaces.pop(old_name)

        self.observation_space = new_obs_space
        self.old_name = old_name
        self.new_name = new_name

    def observation(self, observation:dict):
        observation[self.new_name] = observation.pop(self.old_name)
        return observation

class RewardObs(Wrapper):
    """Make the reward part """

    def __init__(self, env):
        Wrapper.__init__(self, env)
        spaces = self.env.observation_space.spaces
        if "obs_1d" in spaces:
            assert isinstance(spaces["obs_1d"], gym.spaces.Box)
            assert spaces["obs_1d"].dtype == np.float32
            new_space = gym.spaces.Box(-np.inf, np.inf,
                                       shape=tuple(np.array(spaces["obs_1d"].shape) + 1))
        else:
            new_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))
        spaces["obs_1d"] = new_space
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs['vector'] = np.concatenate(
            (obs.get('vector', ()), np.array([reward], dtype=np.float32)),
            axis=-1
        )
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs['vector'] = np.concatenate(
            (obs.get('vector', ()), np.array([0], dtype=np.float32)),
            axis=-1
        )
        return obs
