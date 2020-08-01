from .conf import AgentConf
from Agent.random_network_agent import RandomNetworkAgent

def make(agent_conf: AgentConf,replays)->RandomNetworkAgent:
    if agent_conf.algorithm == "random":
        return RandomNetworkAgent(agent_conf,replays)
    elif agent_conf.algorithm == "sac":
        from .soft_actor_critic_agent import SoftActorCriticAgent
        return SoftActorCriticAgent(agent_conf,replays)