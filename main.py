from franQ.common_utils import AttrDict,time_stamp_str
from pathlib import Path
from franQ import Env, Agent
import multiprocessing as mp
from franQ.Runner.runner import Runner


def main():
    """Setup config and call experiment launcher"""
    #TODO: Allow reading conf from file and setting up argparse for poitning to config file
    global_conf = AttrDict()  # joint global config

    # configure the environment
    env_conf = Env.EnvConf()
    env_conf.suite = "bit_flip"
    env_conf.name = "random-v8"
    global_conf.update(env_conf)  # merge

    # configure the agent
    agent_conf = Agent.AgentConf()
    agent_conf.num_instances = 10
    agent_conf.inference_device = "cpu"
    agent_conf.use_HER = True
    agent_conf.use_nStep_lowerbounds = False
    agent_conf.num_q_networks = 5

    # agent_conf.inference_device = "cpu"
    # agent_conf.replay_size = int(1e4)
    # agent_conf.batch_size = 32
    global_conf.update(agent_conf)

    launch_experiment(global_conf)

import typing as T
def launch_experiment(config: T.Union[Env.EnvConf, Agent.AgentConf]):
    """Launches the experiment"""

    # Make a dummy environment so we can get observation and action space data
    kwargs = {}
    config.log_dir = str(Path(config.log_dir) / time_stamp_str())
    environment = Env.make(config)
    config.obs_space, config.action_space = environment.observation_space, environment.action_space
    if config.use_HER:
        kwargs["compute_reward"] = environment.get_reward_functor()
    import gym
    config.discrete = isinstance(config.action_space,gym.spaces.Discrete)
    del (environment)

    # Launch the experiment
    runner = Runner(config,**kwargs)
    runner.launch()


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
