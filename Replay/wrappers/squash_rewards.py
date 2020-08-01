from typing import Dict
import numpy as np
from .wrapper_base_class import ReplayMemoryWrapper

def _pohlen_transform(x, epsilon=1e-2, pow=0.5):
    """ Reduce variance. Based on Pohlen Transform [arXiv:1805.11593]"""
    return np.sign(x) * (np.power(np.abs(x) + 1, pow) - 1) + epsilon * x


class SquashRewards(ReplayMemoryWrapper):
    """
    Reduce variance in rewards using Pohlen et al's transform [arXiv:1805.11593]
    This usually doesn't change the optimal policy
    """

    def add(self, experience_dict: Dict[str, np.ndarray]):
        experience_dict["reward"] = _pohlen_transform(experience_dict["reward"])
        ReplayMemoryWrapper.add(self, experience_dict)
