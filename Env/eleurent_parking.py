try:
    import highway_env
except ImportError as e:
    Warning("Run: pip install --user git+https://github.com/eleurent/highway-env")
    raise e
import gym
from .conf import EnvConf
from .wrappers import wrapper_base, common, common_image
import numpy as np
import typing as T


class Parking(wrapper_base.ObservationWrapper):
    """Modular transformations over classic OpenAI gym state-vector envs to be compatible with our agents"""

    def __init__(self, conf: EnvConf):
        env = gym.make("parking-v0")
        env.reset()

        # This is the goal oriented obs dict. Let's mod it to be compliant rather than create a whole new dict
        env = common.ObsDictRenameKey(env, old_name="observation", new_name="obs_1d")
        env.observation_space.spaces["obs_1d"] = gym.spaces.Box(
            low=np.min(env.observation_space.spaces["obs_1d"].low).item(),
            high=np.max(env.observation_space.spaces["obs_1d"].high).item(),
            shape=(np.prod(env.observation_space.spaces["obs_1d"].shape).item(),)
        )

        if not isinstance(env.action_space, gym.spaces.Discrete):
            env = common.NormalizeActions(env)

        super().__init__(env)

    def observation(self, observation):
        observation["obs_1d"] = np.reshape(observation["obs_1d"], (-1,))
        return observation

    def get_reward_functor(self) -> T.Callable:
        p: float = 0.5
        reward_weights = self.REWARD_WEIGHTS
        success_goal_reward = self.SUCCESS_GOAL_REWARD

        def compute_reward(achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict):
            """
            Proximity to the goal is rewarded
            We use a weighted p-norm

            :param achieved_goal: the goal that was achieved
            :param desired_goal: the goal that was desired
            :param dict info: any supplementary information
            :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
            :return: the corresponding reward
            """
            reward = -np.power(np.dot(np.abs(achieved_goal - desired_goal), reward_weights), p)
            done = reward > - success_goal_reward
            return reward, done

        return compute_reward
