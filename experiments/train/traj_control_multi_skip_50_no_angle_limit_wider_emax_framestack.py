from franQ.common_utils import AttrDict
from franQ import Env, Agent
import multiprocessing as mp
import typing as T
from experiments.utils.launch_experiment import launch_experiment
import py_ics


def main():
    """Setup config and call experiment launcher"""
    # TODO: Allow reading conf from file and setting up argparse for poitning to config file
    global_conf: T.Union[Env.EnvConf, Agent.AgentConf] = AttrDict()  # joint global config

    # configure the environment
    env_conf = Env.EnvConf()
    env_conf.suite = "traj_control"
    env_conf.name = "traj_control-v1"
    env_conf.render = None
    env_conf.monitor = None
    env_conf.frame_stack_conf.enable = True
    env_conf.frame_stack_conf.num_frames = 4
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

    factory = py_ics.gym_env.envs.JPTrajConFactory()
    factory.use_potential_based_rewards = False
    factory.use_product_reward_components = True
    factory.use_cae_reward = False
    factory.residual = False
    factory.level = 1
    factory.time_limit = int(1.2e5)
    factory.frame_skip = 50
    factory.use_angle_limit = False
    factory.use_random_starts = True


    from pathlib import Path
    import os
    SOURCE_ROOT = Path(r"D:\projects\ics\python_ai4ics\py_ics")
    factory.fmu_base_dir = SOURCE_ROOT / 'FMU_Exports/DSME' / ('win_fmu' if os.name == 'nt' else 'linux_fmu')
    factory.fmu_infinity_path = factory.fmu_base_dir / 'Env_TrajControl_DSME_infty.fmu'
    factory.fmu_param_map_path = factory.fmu_base_dir / 'Env_TrajControl_DSME_paramMap.fmu'
    factory.map_dir_path = SOURCE_ROOT / "Maps/"
    global_conf.env_specific_config = factory

    launch_experiment(global_conf)


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
