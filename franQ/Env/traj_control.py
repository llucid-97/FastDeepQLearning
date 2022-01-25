import itertools
import typing as T
import py_ics
import gym

spaces = gym.spaces
from franQ.Env.wrappers import wrapper_base, common
from franQ.Env.conf import EnvConf
from py_ics.Environments import TrajConFactory


class TrajControl(wrapper_base.Wrapper):
    TIME_LIMITS = [
        int(1e3),
        int(1e4),
        int(1e5),
        int(1e6),
    ]

    def __init__(self, conf: EnvConf):
        version = int(conf.name.split('-v')[-1])
        if conf.env_specific_config is None:
            factory = py_ics.Environments.TrajConFactory()
            factory.time_limit = self.TIME_LIMITS[version]
        else:
            factory = conf.env_specific_config
        assert conf.log_dir is not None
        factory.log_dir = conf.log_dir

        env = factory.make_env()
        env = common.NormalizeActions(env)

        env = common.ObsDict(env, default_key="obs_1d")
        super().__init__(env)

    def get_reward_functor(self) -> T.Callable:
        return self.env.compute_reward


if __name__ == '__main__':
    def mainnnnn():
        conf = EnvConf()
        conf.name = "TrajControl-v0"
        conf.log_dir = './logs'
        factory = TrajConFactory()
        factory.time_limit = int(1e6)
        factory.level = 0
        factory.residual = True
        factory.frame_skip = 1
        factory.use_product_reward_components = True
        conf.env_specific_config = factory
        env = TrajControl(conf, )
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
