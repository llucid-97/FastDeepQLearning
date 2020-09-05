from .async_replay_memory import AsyncReplayMemory
from .replay_memory import ReplayMemory
from . import wrappers

from franQ import Agent as __Agent


def make(conf: __Agent.AgentConf, **kwargs):
    """Helper function to construct the replay memories"""
    # Reader Shards
    shards = [AsyncReplayMemory(
        conf.replay_size, conf.batch_size, conf.temporal_len
    ) for _ in range(conf.num_instances)]

    # Construct optional wrappers around it for modular transformations
    writer_wrappers = shards
    if conf.use_squashed_rewards and not conf.use_HER:
        writer_wrappers = [wrappers.squash_rewards.SquashRewards(r) for r in writer_wrappers]
    if conf.use_HER:
        writer_wrappers = [wrappers.her.HindsightNStepReplay(r, conf.nStep_return_steps, conf.gamma,
                                                             kwargs["compute_reward"]) for r in writer_wrappers]
    if conf.use_nStep_lowerbounds and not conf.use_HER:
        writer_wrappers = [wrappers.nstep_return.NStepReturn(r, conf.nStep_return_steps, conf.gamma) for r in
                           writer_wrappers]
    return shards, writer_wrappers
