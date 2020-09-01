from .conf import AgentConf
from Agent.randomagent import RandomAgent
import gym
def make(agent_conf: AgentConf,replays)->RandomAgent:
    # Sanity checks
    assert isinstance(agent_conf.obs_space,gym.spaces.Dict)
    if "obs_1d" in agent_conf.obs_space.spaces:
        assert len(agent_conf.obs_space.spaces["obs_1d"].shape)==1

    if agent_conf.algorithm.lower() == "random":
        return RandomAgent(agent_conf,replays)
    elif agent_conf.algorithm.lower() == "deep_q_learning":
        from Agent.deepQlearning import DeepQLearning
        return DeepQLearning(agent_conf,replays)