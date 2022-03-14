import numpy as np
import torch
import dataclasses
from collections import deque

def _get_grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class AutoClip():
    def __init__(self, history_len, clip_percentile):
        self.grad_history = deque(maxlen=history_len)
        self.clip_percentile=clip_percentile

    def autoclip_gradient(self,params):
        obs_grad_norm = _get_grad_norm(params)
        self.grad_history.append(obs_grad_norm)
        clip_value = np.percentile(self.grad_history, self.clip_percentile)
        torch.nn.utils.clip_grad_norm_(params, clip_value)

class AutoClipLeaky():
    def __init__(self, history_len, clip_percentile):
        self.grad_history = None
        assert int(history_len)>0
        self.gamma = 1/int(history_len)
        self.clip_percentile=clip_percentile

    def autoclip_gradient(self,params):
        obs_grad_norm = _get_grad_norm(params)
        if self.grad_history is None:
            self.grad_history = obs_grad_norm
        else:
            self.grad_history = (obs_grad_norm * self.gamma) + (self.grad_history * (1-self.gamma))
        clip_value = np.percentile(self.grad_history, self.clip_percentile)
        torch.nn.utils.clip_grad_norm_(params, clip_value)