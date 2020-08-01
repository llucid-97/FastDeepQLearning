import time as _time


class AttrDict(dict):
    """Allow accessing members via getitem for ease of use"""
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def to_dict(self):
        # For serializing (eg to push across process boundaries)
        d = dict()
        d.update(self)
        return d

    def from_dict(self, x):
        self.update(x)
        return self


def time_stamp_str():  # generate a timestamp used for logging
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d___%H-%M-%S")


class TimerSummary:
    """Scope timer that prints to SummaryWriter"""

    def __init__(self, writer, name, group="Timers", step=None):
        from torch.utils.tensorboard import SummaryWriter
        self.writer: SummaryWriter = writer
        self.group, self.name, self.step = group, name, step

    def __enter__(self, ):
        self.start = _time.clock()

    def __exit__(self, *args):
        self.end = _time.clock()
        self.interval = self.end - self.start
        self.writer.add_scalars(self.group, {self.name: self.interval}, self.step)
