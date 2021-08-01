from .conf import EnvConf
from .wrappers.wrapper_base import Wrapper as EnvAPI

def make_mp(env_conf):
    # Makes an env in another process and provides an interface to it
    from .mp_wrapper import MultiProcessingWrapper
    return MultiProcessingWrapper(make,env_conf)

def make(env_conf: EnvConf):
    suite = env_conf.suite.lower()
    if suite == "classic":
        from .classic import Classic  as EnvClass
    elif suite == "classic_pixel":
        from .classic_pixel import ClassicPixel as EnvClass
    elif suite == "eleurent_parking":
        from .eleurent_parking import Parking as EnvClass
    elif suite == "bit_flip":
        from .bitflip import BitFlippingEnv as EnvClass
    elif suite in ("cartpolegoal", "cartpole_goal", "cartpole-goal"):
        from .cartpole_goal import CartPoleGoalEnv as EnvClass
    else:
        raise NotImplementedError(f"Suite {suite} not found!")
    return EnvClass(env_conf)
