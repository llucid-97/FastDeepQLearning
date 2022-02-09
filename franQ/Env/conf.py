from franQ.common_utils import AttrDict
import typing as T


class EnvConf(AttrDict):
    def __init__(self):
        super().__init__()
        self.name = "LunarLanderContinuous-v2"
        self.suite = "classic"
        self.instance_tag:T.Optional[int] = None
        self.num_instances = 2
        self.max_num_episodes = float('inf')
        self.resolution = (84, 84)
        self.render: T.Optional[int] = True
        self.monitor = False # If enabled, captures a video of the rollouts using the render and ffmpeg
        self.artefact_root: T.Optional[str] = None
        self.env_specific_config = None
