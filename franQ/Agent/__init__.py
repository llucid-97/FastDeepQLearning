from .conf import AgentConf
from franQ.Agent.deepQlearning import DeepQLearning
import gym
def make(agent_conf: AgentConf,)->DeepQLearning:
    # Sanity checks
    assert isinstance(agent_conf.obs_space,gym.spaces.Dict)
    if "obs_1d" in agent_conf.obs_space.spaces:
        assert len(agent_conf.obs_space.spaces["obs_1d"].shape)==1

    if agent_conf.algorithm.lower() == "random":
        from franQ.Agent.randomagent import RandomAgent
        return RandomAgent(agent_conf)
    elif agent_conf.algorithm.lower() == "deep_q_learning":
        from franQ.Agent.deepQlearning import DeepQLearning
        return DeepQLearning(agent_conf)