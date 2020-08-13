from common_utils import AttrDict,time_stamp_str
from pathlib import Path
import Agent, Env
import multiprocessing as mp
from Runner.runner import Runner


def main():
    """Setup config and call experiment launcher"""
    #TODO: Allow reading conf from file and setting up argparse for poitning to config file
    global_conf = AttrDict()  # joint global config

    # configure the environment
    env_conf = Env.EnvConf()
    env_conf.suite = "classic"
    env_conf.name = "LunarLanderContinuous-v2"
    global_conf.update(env_conf)  # merge

    # configure the agent
    agent_conf = Agent.AgentConf()
    agent_conf.num_instances = 4
    agent_conf.num_critic_predictions = 10
    agent_conf.use_double_q = False
    agent_conf.algorithm = "sac"
    # agent_conf.use_sde = True
    global_conf.update(agent_conf)

    launch_experiment(global_conf)


def launch_experiment(config: Agent.AgentConf):
    """Launches the experiment"""

    # Make a dummy environment so we can get observation and action space data
    config.log_dir = str(Path(config.log_dir) / time_stamp_str())
    environment = Env.make(config)
    config.obs_space, config.action_space = environment.observation_space, environment.action_space
    del (environment)

    # Launch the experiment
    runner = Runner(config)
    runner.launch()


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
