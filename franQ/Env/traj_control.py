from collections import OrderedDict
from typing import Optional, Union, Callable
import py_ics
import numpy as np
import gym
from gym import wrappers
import math

spaces = gym.spaces
from .wrappers import wrapper_base, common
from .conf import EnvConf


class TrajControl(wrapper_base.Wrapper):
    TIME_LIMITS = [
        int(1e3),
        int(1e4),
        int(1e5),
        int(1e6),
    ]

    def __init__(self, conf: EnvConf,factory=None):
        version = int(conf.name.split('-v')[-1])
        factory = py_ics.Environments.TrajConFactory() if factory is None else factory
        factory.time_limit = self.TIME_LIMITS[version]
        assert conf.logdir is not None
        factory.log_dir = conf.logdir

        env = factory.make_env()
        env = common.NormalizeActions(env)

        env = common.ObsDict(env, default_key="obs_1d")
        super().__init__(env)

    def get_reward_functor(self) -> Callable:
        return self.env.compute_reward


if __name__ == '__main__':
    def main():
        conf = EnvConf()
        conf.name = "TrajControl-v0"
        conf.logdir = './logs'
        env = TrajControl(conf)
        while True:
            obs = env.reset()
            done = False

            while not done:
                action = env.action_space.sample()
                assert -1 <= action <= 1
                assert "obs_1d" in obs
                obs, reward, done, info = env.step(action)
                from pprint import pprint
                pprint(info)


    main()
