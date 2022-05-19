from franQ.common_utils import AttrDict
from franQ import Env, Agent
import multiprocessing as mp
import typing as T
from experiments.utils.launch_experiment import launch_experiment
import py_ics


def main():
    """
    This sets up the FMU Full benchmark o
    :return:
    """
    # TODO: Allow reading conf from file and setting up argparse for poitning to config file
    global_conf: T.Union[Env.EnvConf, Agent.AgentConf] = AttrDict()  # joint global config

    # configure the environment
    env_conf = Env.EnvConf()
    env_conf.suite = "traj_control-v2"
    env_conf.render = None
    env_conf.monitor = None
    env_conf.frame_stack_conf.enable = True
    env_conf.frame_stack_conf.num_frames = 10
    env_conf.frame_stack_conf.exponential_mode = True
    global_conf.update(env_conf)  # merge

    # configure the agent
    agent_conf = Agent.AgentConf()
    agent_conf.num_instances = 5
    agent_conf.inference_device = "cpu"
    agent_conf.use_nStep_lowerbounds = True
    agent_conf.num_critics = 5

    # NOTE: Fewer layers ===> Faster training.
    agent_conf.enc1d_hidden_dims = []
    agent_conf.pi_hidden_dims = [256]
    agent_conf.critic_hidden_dims = [256, 256]
    agent_conf.init_log_alpha = 0
    agent_conf.replay_size = int(1e5)
    global_conf.update(agent_conf)

    from franQ.Env.traj_control_v2 import TrajControlWrapperConf
    factory = TrajControlWrapperConf()
    # Level / scenario select
    factory.level_select_policy = "cycle"
    factory.use_fixed_scenarios = True
    factory.level = 1

    # Objective
    factory.use_potential_based_rewards = False
    factory.use_product_reward_components = True
    factory.use_cae_reward = False
    factory.time_limit = int(1.2e5)
    factory.frame_skip = 30
    factory.use_random_starts = True

    # action space
    factory.residual = False
    factory.use_angle_limit = True


    global_conf.env_specific_config = factory
    launch_experiment(global_conf)


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
