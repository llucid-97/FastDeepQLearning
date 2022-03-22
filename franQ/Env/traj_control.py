import itertools
import typing as T
import py_ics
import gym

spaces = gym.spaces
from franQ.Env.wrappers import wrapper_base, common
from franQ.Env.conf import EnvConf
from py_ics.gym_env.envs import JPTrajConFactory


class TrajControlWrapper(wrapper_base.Wrapper):

    def __init__(self, conf: EnvConf):
        self.version = version = int(conf.name.split('-v')[-1])


        if conf.env_specific_config is None:
            factory = JPTrajConFactory()
        else:
            factory = conf.env_specific_config

        if version > 0:
            # This is designed for multi env generators. It maps each to a specific level.
            self.idx = idx = conf.instance_tag
            self.num_instances = conf.num_instances
            if conf.instance_tag is None:
                import logging
                logging.warning("TrajControl-v1 REQUIRES INSTANCE TAG! ASSUMING SET TO 0!")
                idx = 0
            factory.level = (idx % 5)

        assert conf.log_dir is not None
        factory.log_dir = conf.log_dir
        self.factory = factory
        self._init_actual(conf)

    def _init_actual(self,conf:EnvConf):
        env = self.factory.make_env()
        env = common.NormalizeActions(env)
        if conf.frame_stack_conf.enable:
            env = common.FrameStack(env,k=conf.frame_stack_conf.num_frames)
        env = common.ObsDict(env, default_key="obs_1d")
        super().__init__(env)

    def reset(self, **kwargs):
        if self.version > 1:
            # Cycle environments on reset
            super().reset() # to allow episodic logging
            self.factory.level = (self.factory.level + self.num_instances) % self.factory.num_levels
            self._init_actual()
        return super().reset()


    def get_reward_functor(self) -> T.Callable:
        return self.env.compute_reward


if __name__ == '__main__':
    def mainnnnn():
        conf = EnvConf()
        conf.name = "TrajControl-v0"
        conf.log_dir = './logs'
        factory = JPTrajConFactory()
        factory.time_limit = int(1e6)
        factory.level = 0
        factory.residual = True
        factory.frame_skip = 1
        factory.use_product_reward_components = True
        conf.env_specific_config = factory
        env = TrajControlWrapper(conf, )
        import itertools
        for episode_num in itertools.count():
            obs = env.reset()
            done = False
            score = 0
            info = {}
            while not done:
                action = env.action_space.sample()
                assert -1 <= action <= 1
                assert "obs_1d" in obs
                obs, reward, done, info = env.step(0.0)
                score += reward
                from numbers import Number
                import numpy as np
                np.set_printoptions(precision=2)
                print("\r" + "\t".join(
                    [f"{k}=" + (f"{v:.2f} |" if isinstance(v, Number) else f"{v}") for k, v in info.items()]),
                      end=""
                      )
                import time
                # time.sleep(1)
            print()
            print(f"Episode {episode_num} terminated!")
            print(f"score={score}")
            print(info)
            print()


    mainnnnn()
