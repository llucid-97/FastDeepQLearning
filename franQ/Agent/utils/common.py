import torch


def squash_variance(x, epsilon=1e-2, pow=0.5):
    """ Reduce variance. Based on Pohlen Transform [arXiv:1805.11593]"""
    x = torch.sign(x) * (torch.pow(torch.abs(x) + 1, pow) - 1) + epsilon * x
    return x


def soft_update(target, source, tau):
    if target is source: return
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source, *args, **kwargs):
    if target is source: return
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
