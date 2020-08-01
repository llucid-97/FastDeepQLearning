from .conf import EnvConf
from .gym_wrapper import Wrapper as EnvAPI

def make_mp(env_conf):
    from .multiprocessing import MultiProcessingWrapper
    return MultiProcessingWrapper(make,env_conf)

def make(env_conf: EnvConf):
    if env_conf.suite.lower() == "classic":
        from .classic import Classic
        env = Classic(env_conf)

    return env
