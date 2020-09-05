from franQ import Replay
from franQ.Replay.async_replay_memory import AsyncReplayMemory
from franQ.Replay.replay_memory import ReplayMemory
import numpy as np


def test_nstep_return():
    discount = 0.99
    n_step = 1000
    replay = ReplayMemory(1001, batch_size=128, temporal_len=1)
    replay = franQ.Replay.wrappers.NStepReturn(replay, n_step=n_step, discount=discount)

    # put 1000 entries
    for i in range(n_step):
        experience = {
            "reward": float(i == (n_step - 1)),
            "episode_done": i == (n_step - 1),
            "step": i,
        }
        replay.add(experience)

    for j in range(100):
        s = replay.sample()
        assert np.allclose(s["mc_return"], discount ** (n_step - 1 - s["step"]))


def test_size():
    import numpy as np
    kwargs = {
        "maxlen": int(5e3),
        "batch_size": 256,
        "temporal_len": 10,
    }
    r = AsyncReplayMemory(**kwargs)
    # Check that while not full, the size is same as index
    for i in range(kwargs["maxlen"] * 2):
        r.add({"obs": np.random.uniform(size=[10]), "action": 2})
        if i < kwargs["maxlen"]:
            assert len(r) == (i + 1)

        assert len(r) >= (1 + i) - kwargs["maxlen"]
        assert len(r) <= i + 1
    # Check that when we overflow, the size doesn't overrun
    for i in range(kwargs["maxlen"]):
        r.add({"obs": np.random.uniform(size=[10]), "action": 2})

    assert len(r) == kwargs["maxlen"]
    return True


def test_temporal_consistency():
    import numpy as np
    import torch
    from franQ.Replay.wrappers import torch_dataloader
    kwargs = {
        "maxlen": int(5e3),
        "batch_size": 256,
        "temporal_len": 10,
    }
    r = AsyncReplayMemory(**kwargs)

    # Check that while not full, the size is same as index
    obs_size = 10
    for i in range(kwargs["maxlen"] // 2):
        r.add({"obs": np.ones([obs_size]) * i, "action": 2})
    # Check that when we overflow, the size doesn't overrun

    l = torch_dataloader.TorchDataLoader(r)
    for _ in range(100):
        experience_dict = l.temporal_sample()
        obs: torch.Tensor = experience_dict["obs"]
        assert tuple(obs.shape) == (kwargs["temporal_len"], kwargs["batch_size"], obs_size)
        is_contiguous = obs[1:] == (obs[:-1] + 1)
        assert is_contiguous.all()
    return True


if __name__ == '__main__':
    def run_tests():
        test_nstep_return()
        test_size()
        test_temporal_consistency()
        print("passed!")
        exit(0)


    run_tests()
