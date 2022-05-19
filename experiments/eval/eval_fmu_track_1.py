from franQ.common_utils import AttrDict
from franQ import Env, Agent
import multiprocessing as mp
from experiments.utils.launch_experiment import evaluate_experiment
import py_ics

def main(eval_dir):
    from pathlib import Path
    assert Path(eval_dir).exists(), "Please specify a path for an existing experiment to evaluate!"
    """Setup config and call experiment launcher"""
    # TODO: Allow reading conf from file and setting up argparse for poitning to config file
    global_conf = AttrDict()  # joint global config

    # configure the environment
    env_conf = Env.EnvConf()
    env_conf.suite = "traj_control"
    env_conf.name = "traj_control-v0"
    env_conf.render = True
    env_conf.monitor = None
    global_conf.update(env_conf)  # merge

    # configure the agent
    agent_conf = Agent.AgentConf()
    agent_conf.num_instances = 1
    agent_conf.inference_device = "cpu"
    agent_conf.use_nStep_lowerbounds = True
    agent_conf.num_critics = 5

    # NOTE: Fewer layers ===> Faster training.
    agent_conf.enc1d_hidden_dims = []
    agent_conf.pi_hidden_dims = [256]
    agent_conf.critic_hidden_dims = [256, 256]
    agent_conf.init_log_alpha = 0

    global_conf.update(agent_conf)

    factory = py_ics.gym_custom.envs.JPTrajConFactory()
    factory.use_potential_based_rewards = False
    factory.use_product_reward_components = True
    factory.use_cae_reward = True
    factory.residual = False
    factory.level = 1
    factory.time_limit = int(1e7)
    factory.frame_skip = 50
    factory.async_render = True
    global_conf.env_specific_config = factory

    evaluate_experiment(
        global_conf,
        eval_dir,
        episodes=1,
        worker_seeds=[0, ]
    )


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main(
    r"D:\projects\ics\python_ai4ics\py_ics\repos\fastdeepqlearning\experiments\logs\2022-02-22___12-12-18traj_control_traj_control-v1"
    )
