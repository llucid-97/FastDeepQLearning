import time
import time as _time

_time.clock = time.time
try:
    import pyjion

    pyjion.enable()
except ImportError:
    pass


class AttrDict(dict):
    """Allow accessing members via getitem for ease of use"""
    __setattr__ = dict.__setitem__

    def __getattr__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError as e:
            raise AttributeError(e)


def kill_proc_tree(pid, including_parent=True):
    import psutil
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        psutil.wait_procs(children, timeout=5)
        if including_parent:
            parent.kill()
            parent.wait(5)
    except psutil.NoSuchProcess:
        pass


def numpy_set_print_decimal_places(num_decimal_places=2):
    import numpy as np
    float_formatter = f"{{:.{num_decimal_places}f}}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})


def time_stamp_str():  # generate a timestamp used for logging
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d___%H-%M-%S")


class TimerTB:
    """Scope timer that prints to SummaryWriter"""
    CLASS_ENABLE_SWITCH = True  # a kill switch to conveniently enable/disable logging in 1 central place

    def __init__(self, writer, name, group="Timers", step=None, force_enable=False):
        from torch.utils.tensorboard import SummaryWriter
        self.writer: SummaryWriter = writer
        self.group, self.name, self.step = group, name, step
        self.force_enable = force_enable

    def __enter__(self, ):
        self.start = _time.clock()

    def __exit__(self, *args):
        self.end = _time.clock()
        self.interval = self.end - self.start
        if self.CLASS_ENABLE_SWITCH or self.force_enable:
            self.writer.add_scalars(self.group, {self.name: self.interval}, self.step)


class LeakyIntegrator:
    def __init__(self, time_window=100, error_thresh=0.01):
        import numpy as np
        self.gamma = np.exp(np.log(error_thresh) / time_window)
        self.leaky_x = 0

    def __call__(self, x):
        self.leaky_x = (self.gamma * self.leaky_x) + ((1 - self.gamma) * x)
        return self.leaky_x


try:
    pyjion.disable()
except:
    pass
