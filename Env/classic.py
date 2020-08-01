import gym
from .conf import EnvConf
from .gym_wrapper import Wrapper
class Classic(Wrapper):
    def __init__(self,env_conf:EnvConf):
        env = gym.make(env_conf.name)
        self.conf = env_conf
        super().__init__(env)