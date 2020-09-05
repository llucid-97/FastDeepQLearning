import gym
import numpy as np
import cv2

from .wrapper_base import ObservationWrapper


class Nhwc2Nchw(ObservationWrapper):
    """Move channel axis to front (for compliance with torch-style convolutions)"""

    def __init__(self, env):
        super(Nhwc2Nchw, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.moveaxis(observation, -1, 0)


class ForcePixelObs(ObservationWrapper):
    """Forces observations to be raw pixels by replacing them with the rgb_array render."""

    def __init__(self, env, res=(84, 84)):
        super(ForcePixelObs, self).__init__(env)
        self._res = res
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=res + (3,), dtype=np.float32)

    def observation(self, observation):
        obs: np.ndarray = cv2.resize(self.env.render(mode="rgb_array"), self._res)
        obs = obs.astype(np.float) / 255
        return obs


class ResizeImage(ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(0, 255, (height, width, self.observation_space.shape[2]), np.uint8)

    def observation(self, frame):
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)


class ColorTransform(ObservationWrapper):
    def __init__(self, env: gym.Env, code=cv2.COLOR_RGB2HSV):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        ObservationWrapper.__init__(self, env)
        self.code = code

        # determine the new obs space dims.
        frame: np.ndarray = env.observation_space.sample()
        frame = self.observation(frame)
        self.observation_space = gym.spaces.Box(env.observation_space.high,
                                                env.observation_space.low,
                                                frame.shape,
                                                frame.dtype)

    def observation(self, frame):
        return cv2.cvtColor(frame, self.code)
