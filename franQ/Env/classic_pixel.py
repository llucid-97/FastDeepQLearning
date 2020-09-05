import gym
from .conf import EnvConf
from .wrappers import wrapper_base, common, common_image


class ClassicPixel(wrapper_base.Wrapper):
    """Modular transformations over classic OpenAI gym state-vector envs to be compatible with our agents.
    Uses the rgb_array render as the observations
    """

    def __init__(self, conf: EnvConf):
        env = gym.make(conf.name)
        env = common_image.ResizeImage(env, *conf.resolution)
        env = common_image.Nhwc2Nchw(env)
        env = common.ObsDict(env, "obs_2d")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            env = common.NormalizeActions(env)
        super().__init__(env)
