from franQ.common_utils import AttrDict
from franQ import Env, Agent
import multiprocessing as mp
from experiments.utils.launch_experiment import launch_experiment


def main():
    """Setup config and call experiment launcher"""
    # TODO: Allow reading conf from file and setting up argparse for poitning to config file
    global_conf = AttrDict()  # joint global config

    # configure the environment
    env_conf = Env.EnvConf()
    env_conf.suite = "classic_goal"
    env_conf.name = "Acrobot-v1"
    env_conf.render = False
    env_conf.monitor = True
    global_conf.update(env_conf)  # merge

    # configure the agent
    agent_conf = Agent.AgentConf()
    agent_conf.num_instances = 3
    agent_conf.inference_device = "cpu"
    agent_conf.use_nStep_lowerbounds = True
    agent_conf.use_HER = True
    agent_conf.num_critics = 5

    # NOTE: Fewer layers ===> Faster training.
    agent_conf.enc1d_hidden_dims = [256]
    agent_conf.pi_hidden_dims = [256]
    agent_conf.critic_hidden_dims = [256, 256]
    agent_conf.init_log_alpha = 0

    global_conf.update(agent_conf)

    launch_experiment(global_conf)


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
