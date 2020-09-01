import gym
import cv2
import multiprocessing as mp
import os
import numpy as np
import psutil


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

    def __del__(self):
        def kill_proc_tree(pid, including_parent=True):
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.kill()
                gone, still_alive = psutil.wait_procs(children, timeout=5)
                if including_parent:
                    parent.kill()
                    parent.wait(5)
            except psutil.NoSuchProcess:
                print(f"Forced {type(self)} to kill child process")

        # kill_proc_tree(self.render_proc.pid)
