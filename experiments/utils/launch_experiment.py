import typing as T
from franQ import Env,Agent,Runner,common_utils
from pathlib import Path

def launch_experiment(config: T.Union[Env.EnvConf, Agent.AgentConf]):
    """Launches the experiment"""

    # Make a dummy environment so we can get observation and action space data
    kwargs = {}
    config.log_dir = str(Path(config.log_dir) / (common_utils.time_stamp_str() + f"{config.suite}_{config.name}"))
    config.artefact_root = str(Path(config.log_dir) / "artefacts")
    import copy
    c = copy.copy(config)
    c.monitor = False
    eg_env = Env.make(config)
    config.obs_space, config.action_space = eg_env.observation_space, eg_env.action_space
    if config.use_HER:
        kwargs["compute_reward"] = eg_env.get_reward_functor()
    import gym
    config.discrete = isinstance(config.action_space,gym.spaces.Discrete)
    del (eg_env)

    # Launch the experiment
    runner = Runner.Runner(config,**kwargs)
    runner.launch()