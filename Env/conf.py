from common_utils import AttrDict
class EnvConf(AttrDict):
    def __init__(self):
        super().__init__()
        import numpy as np

        self.name = "LunarLanderContinuous-v2"
        self.suite = "classic"
        self.max_num_episodes = np.inf
