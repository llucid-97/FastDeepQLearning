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
    write_heads = read_heads = shards

    if conf.use_nStep_lowerbounds:
        if conf.her_mode == "vmap":
            write_heads = [wrappers.nstep_return_vmap.NStepReturnVmap(r, conf.nStep_return_steps, conf.gamma) for r
                           in write_heads]
        else:
            write_heads = [wrappers.nstep_return.NStepReturn(r, conf.nStep_return_steps, conf.gamma) for r in
                           write_heads]
    if conf.use_squashed_rewards and not conf.use_HER:
        write_heads = [wrappers.squash_rewards.SquashRewards(r) for r in write_heads]
    if conf.use_HER:
        if conf.her_mode == "vmap":
            write_heads = [wrappers.her_vmap.HindsightVmapWrite(r, kwargs["compute_reward"], ) for r in write_heads]
            read_heads = [wrappers.her_vmap.HindsightVmapRead(r) for r in read_heads]
        else:
            write_heads = [wrappers.her.HindsightNStepReplay(r, kwargs["compute_reward"],
                                                             mode=conf.her_mode) for r in write_heads]

    return read_heads, write_heads
