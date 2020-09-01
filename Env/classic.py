import gym
from .conf import EnvConf
from .wrappers import wrapper_base,common,common_image

class Classic(wrapper_base.Wrapper):
    """Modular transformations over classic OpenAI gym state-vector envs to be compatible with our agents"""
    def __init__(self, conf:EnvConf):
        env = gym.make(conf.name)
        if conf.force_pixel:
            env = common_image.ForcePixelObs(env, res=conf.resolution)
            env = common_image.Nhwc2Nchw(env)
            env = common.ObsDict(env, "obs_2d")
        else:
            env = common.ObsDict(env, "obs_1d")
        if not isinstance(env.action_space,gym.spaces.Discrete):
            env = common.NormalizeActions(env)
        super().__init__(env)