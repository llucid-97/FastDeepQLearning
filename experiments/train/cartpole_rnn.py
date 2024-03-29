from franQ.common_utils import AttrDict
from franQ import Env, Agent
from torch import multiprocessing as mp
from experiments.utils.launch_experiment import launch_experiment


def main():
    """Setup config and call experiment launcher"""
    # TODO: Allow reading conf from file and setting up argparse for poitning to config file
    global_conf = AttrDict()  # joint global config

    # configure the environment
    env_conf = Env.EnvConf()
    env_conf.suite = "classic"
    env_conf.name = "CartPole-v1"
    env_conf.render = None
    env_conf.monitor = None
    global_conf.update(env_conf)  # merge

    # configure the agent
    agent_conf = Agent.AgentConf()
    agent_conf.num_instances = 3
    agent_conf.inference_device = "cpu"
    agent_conf.use_nStep_lowerbounds = True
    agent_conf.num_critics = 5

    # NOTE: Fewer layers ===> Faster training.
    agent_conf.encoder_conf.joiner_mode = agent_conf.encoder_conf.JoinerModeEnum.gru
    agent_conf.encoder_conf.use_burn_in = True
    agent_conf.encoder_conf.rnn_latent_state_training_mode = agent_conf.encoder_conf.RnnLatentStateTrainMode.store

    agent_conf.pi_hidden_dims = [256]
    agent_conf.critic_hidden_dims = [256, 256]
    agent_conf.init_log_alpha = 0

    global_conf.update(agent_conf)

    launch_experiment(global_conf)


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
