from .conf import EnvConf
from .wrappers.wrapper_base import Wrapper as EnvAPI


def make_mp(env_conf):
    # Makes an env in another process and provides an interface to it
    from .mp_wrapper import MultiProcessingWrapper
    return MultiProcessingWrapper(make, env_conf)


def make(conf: EnvConf):
    suite = conf.suite.lower()
    if suite == "classic":
        from .classic import Classic as EnvClass
    elif suite == "classic_longer":
        from .classic_longer import Classic as EnvClass
    elif suite == "classic_pixel":
        from .classic_pixel import ClassicPixel as EnvClass
    elif suite == "eleurent_parking":
        from .eleurent_parking import Parking as EnvClass
    elif suite == "bit_flip":
        from .bitflip import BitFlippingEnv as EnvClass
    elif suite in ("classic_goal",):
        from franQ.Env.classic_control_goal import ClassicGoalEnv as EnvClass
    elif suite in ("traj_control",):
        from franQ.Env.traj_control import TrajControlWrapper as EnvClass
    elif suite in ("traj_control-v2",):
        from franQ.Env.traj_control_v2 import TrajControlV2Wrapper as EnvClass
    else:
        raise NotImplementedError(f"Suite {suite} not found!")

    env = EnvClass(conf)
    if conf.monitor:
        from gym.wrappers import Monitor
        from pathlib import Path
        import uuid
        env = Monitor(env, str(Path(conf.log_dir) / "artefacts" / f"monitor_{conf.instance_tag}_{uuid.uuid4()}"))
    return env
