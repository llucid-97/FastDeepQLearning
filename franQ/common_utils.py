import time
import time as _time

_time.clock = time.time


class PyjionJit:
    """A configurable decorator and context manager for pyjion to make it easy to use or not to use :)"""
    GLOBAL_ENABLE = False

    def __init__(self, func=None, *, exclude=False):
        self.func = func
        self.exclude = exclude

        if self.func is not None:
            import functools
            functools.update_wrapper(self, self.func)

    @staticmethod
    def wrapper(self, *args, **kwargs):
        with self:
            value = self.func(*args, **kwargs)
        return value

    def __call__(self, *args, **kwargs):

        if self.func is None:
            # @PyjionJit(**kwargs) case
            self.func = args[0]  # no func args in init. So func is getting put here during decorator call
            # import functools
            # functools.update_wrapper(self.wrapper, self.func)
            return self.wrapper  # wrapper gets func from member
        else:
            # @PyjionJit case
            # @decorator has already been called and put the functor in self.func
            # this call is the wrapper
            return self.wrapper(*args, **kwargs)

    def __enter__(self):
        if not self.GLOBAL_ENABLE:
            return
        import pyjion
        if self.exclude:
            self.prior_state = pyjion.enable()
            pyjion.disable()
        else:
            pyjion.enable()

    def __exit__(self, *args):
        if not self.GLOBAL_ENABLE:
            return
        import pyjion
        if self.exclude:
            pyjion.enable()  # we're assuming here that the only reason you'd use the ex
        else:
            pyjion.disable()


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


class Timer():
    """Scope timer that prints to SummaryWriter"""
    CLASS_ENABLE_SWITCH = True  # a kill switch to conveniently enable/disable logging in 1 central place

    def __init__(self, name, ):
        self.name = name

    def __enter__(self, ):
        self.start = _time.clock()

    def __exit__(self, *args):
        self.end = _time.clock()
        self.interval = self.end - self.start
        self.print_fn()

    def print_fn(self):
        if self.CLASS_ENABLE_SWITCH:
            print(f"[Timer] {self.name} took {self.interval}s")


class TimerTB(Timer):
    """Scope timer that prints to SummaryWriter"""

    def __init__(self, writer, name, group="Timers", step=None, force_enable=False, log_every=50):
        from torch.utils.tensorboard import SummaryWriter
        self.writer: SummaryWriter = writer
        self.group, self.step = group, step
        self.force_enable = force_enable
        self.log_every = log_every
        super().__init__(name)

    def print_fn(self):
        if self.CLASS_ENABLE_SWITCH or self.force_enable:
            if (self.step is not None) and (0 == (self.step%self.log_every)):
                self.writer.add_scalars(self.group, {self.name: self.interval}, self.step)


class LeakyIntegrator:
    def __init__(self, time_window=100, error_thresh=0.01):
        import numpy as np
        self.gamma = np.exp(np.log(error_thresh) / time_window)
        self.leaky_x = 0

    def __call__(self, x):
        self.leaky_x = (self.gamma * self.leaky_x) + ((1 - self.gamma) * x)
        return self.leaky_x
