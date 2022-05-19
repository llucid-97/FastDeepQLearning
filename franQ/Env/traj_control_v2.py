import itertools
import typing as T
import py_ics
import gym
import logging
import random

from franQ.Env.wrappers import wrapper_base, common
from franQ.Env.conf import EnvConf, AttrDict
from py_ics.gym_env.envs import FullParametricFMUTrajControlFactory


class TrajControlWrapperConf(FullParametricFMUTrajControlFactory):
    def __init__(self):
        super().__init__()
        self.level_select_policy: T.Literal["fixed", "map_idx_to_level", "cycle", "random"] = "fixed"


class TrajControlV2Wrapper(wrapper_base.Wrapper):

    def __init__(self, conf: EnvConf):
        factory = TrajControlWrapperConf() if conf.env_specific_config is None else conf.env_specific_config

        # This is designed for multi env generators. It maps each to a specific level.
        self.idx = idx = conf.instance_tag
        self.num_instances = conf.num_instances
        if conf.instance_tag is None:
            logging.warning(__file__)
            logging.warning("TrajControl-v1 REQUIRES INSTANCE TAG! ASSUMING SET TO 0!")
            self.idx = idx = 0

        if factory.level_select_policy in ["map_idx_to_level", "cycle"]:
            factory.level = (idx % factory.num_levels)

        assert conf.log_dir is not None
        factory.log_dir = conf.log_dir
        self.factory: TrajControlWrapperConf = factory
        self.preproc_options = {  # need to extract this here so we aren't keeping copies of unserializable state
            "frame_stack_conf": conf.frame_stack_conf,
        }
        self._init_actual(self.preproc_options)

    def _init_actual(self, conf: dict):
        conf: EnvConf = AttrDict(conf)
        env = self.factory.make_env()
        env = self.get_preprocessing_stack(conf, env)
        super().__init__(env)

    @staticmethod
    def get_preprocessing_stack(conf, env):
        env = common.NormalizeActions(env)
        if conf.frame_stack_conf.enable:
            env = common.FrameStack(env, k=conf.frame_stack_conf.num_frames)
        env = common.ObsDict(env, default_key="obs_1d")
        return env

    def reset(self, **kwargs):
        if self.factory.level_select_policy == "cycle": # Cycle environments on reset
            super().reset()  # to allow episodic logging
            self.factory.level = (self.factory.level + self.num_instances) % self.factory.num_levels
            self._init_actual(self.preproc_options)
        elif self.factory.level_select_policy == "random":

            super().reset()  # to allow episodic logging
            self.factory.level = random.randint(0,self.factory.num_levels-1)
            self._init_actual(self.preproc_options)
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
