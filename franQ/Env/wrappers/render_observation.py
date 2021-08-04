import gym
import cv2
import multiprocessing as mp
import os
import numpy as np


def _render(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    cv2.imshow(str(os.getpid()) + "_render", img)
    cv2.waitKey(1)


def _async_callback(q: mp.Queue):
    while True:
        img = q.get()
        _render(img)


class RenderObservation(gym.Wrapper):
    """
    Renders the observation as an image
    """

    def __init__(self, env: gym.Env, asynch=True):
        super(RenderObservation, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.asynch = asynch
        self.last_obs = env.observation_space.sample()

        if asynch:
            self.frame_q = mp.Queue(maxsize=3)
            self.render_proc = mp.Process(target=_async_callback, args=(self.frame_q,))
            self.render_proc.start()

    def reset(self, **kwargs):
        obs = super(RenderObservation, self).reset()
        self.last_obs = obs
        return obs

    def step(self, action):
        tup = super(RenderObservation, self).step(action)
        self.last_obs = tup[0]
        return tup

    def render(self, mode="human", **kwargs):
        if self.asynch:
            if not self.frame_q.full():
                self.frame_q.put_nowait(self.last_obs)
        else:
            _render(self.last_obs)
