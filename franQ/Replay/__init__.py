from .async_replay_memory import AsyncReplayMemory
from .replay_memory import ReplayMemory
from . import wrappers

from franQ import Agent as __Agent


def make(conf: __Agent.AgentConf, **kwargs):
    """Helper function to construct the replay memories"""
    # Reader Shards
    from pathlib import Path
    import uuid
    shards = [AsyncReplayMemory(
        int(conf.replay_size), conf.batch_size, conf.temporal_len,
        log_dir=Path(conf.log_dir) / "replay" / f"{uuid.uuid4()}"
    ) for _ in range(conf.num_instances)]

    # Construct optional wrappers around it for modular transformations
    writer_wrappers = shards
    if conf.use_HER:
        writer_wrappers = [wrappers.her.HindsightNStepReplay(r,kwargs["compute_reward"],
                                                             mode=conf.her_mode) for r in writer_wrappers]
    if conf.use_squashed_rewards and not conf.use_HER:
        writer_wrappers = [wrappers.squash_rewards.SquashRewards(r) for r in writer_wrappers]

    if conf.use_nStep_lowerbounds:
        writer_wrappers = [wrappers.nstep_return.NStepReturn(r, conf.nStep_return_steps, conf.gamma) for r in
                           writer_wrappers]
    return shards, writer_wrappers
