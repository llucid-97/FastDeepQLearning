from Replay.replay_memory import ReplayMemory
from Replay.async_replay_memory import AsyncReplayMemory

def temporal_check():
    import numpy as np
    size = 100
    r = ReplayMemory(size)
    for i in range(1000):
        r.add({"obs": np.random.uniform(size=[10]), "action": 2})
    s = r.temporal_sample(5, 3)
    print(s)


def sanity_check():
    size = 100
    r = ReplayMemory(size)
    for i in range(1000):
        r.add(i)
        s = r.sample(10)
        for j in s:
            assert j >= i - size
            assert j <= i
    print("sanity check passed!")


def attr_check():
    size = 100
    r = ReplayMemory(size)

    for i in range(1000):
        r.add(i)
        s = r.sample(10)

    assert r.maxlen
    assert r[0]
    assert len(r) == size
    print("attr check passed!")


def wrap_around_check():
    size = 100
    r = ReplayMemory(size)
    for i in range(size + 1):
        r.add(i)
        s = r.sample(10)
    assert r[0] > r[1]
    print("wrap check passed!")


def async_replay_check():
    maxlen = 10
    batch = 5
    temporal = 3
    replay = AsyncReplayMemory(maxlen, batch, temporal)

    for i in range(maxlen):
        replay.add({"obs": i})

    for _ in range(maxlen):
        s = replay.temporal_sample(batch, temporal)
        assert len(s) == batch
        assert len(s[0]) == temporal


if __name__ == '__main__':
    async_replay_check()
    temporal_check()
    sanity_check()
    attr_check()
    wrap_around_check()
