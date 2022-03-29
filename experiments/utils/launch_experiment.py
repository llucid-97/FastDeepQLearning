import typing as T, copy, os
from pathlib import Path

import gym, numpy as np

import franQ.Agent
from franQ import Env, Agent, Runner, common_utils


def launch_experiment(config: T.Union[Env.EnvConf, Agent.AgentConf]):
    """Launches the experiment"""

    # Make a dummy environment so we can get observation and action space data
    kwargs = {}
    config.log_dir = str(Path(config.log_dir) / (common_utils.time_stamp_str() + f"{config.suite}_{config.name}"))
    config.artefact_root = str(Path(config.log_dir) / "artefacts")
    c = copy.copy(config)
    c.monitor = False
    eg_env = Env.make(config)
    config.obs_space, config.action_space = eg_env.observation_space, eg_env.action_space
    if config.use_HER:
        kwargs["compute_reward"] = eg_env.get_reward_functor()
    config.discrete = isinstance(config.action_space, gym.spaces.Discrete)
    del (eg_env)

    # Launch the experiment
    runner = Runner.Runner(config, **kwargs)
    runner.launch()


def evaluate_policy(
        model_path: T.Union[str, Path],
        env_generator: T.Callable[..., gym.Env],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        log_dir=None,
        render: bool = False,
        callback: T.Optional[T.Callable[[T.Dict[str, T.Any], T.Dict[str, T.Any]], None]] = None,
        reward_threshold: T.Optional[float] = None,
        return_episode_rewards: bool = False,
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    if any([x is not None for x in (callback, reward_threshold)]):
        raise NotImplementedError("Not yet at feature parity with stable-baselines-3. TODO!")
    # Launch the experiment

    rollout_data = Runner.Evaluator()(model_path,
                                      num_episodes=n_eval_episodes,
                                      parallel_instance_seeds=[0],
                                      env_generator=env_generator,
                                      deterministic=deterministic,
                                      log_dir_override=log_dir,
                                      render=render,

                                      )

    scores = [d["score"] for d in rollout_data]
    steps = [d["step"] for d in rollout_data]
    if return_episode_rewards:
        return scores, steps
    # mean and std of score
    return np.mean(scores), np.std(scores)


def evaluate_experiment(config: T.Union[Env.EnvConf, Agent.AgentConf], experiment_log_dir, episodes, seeds,
                        score_reduction_fn=np.min):
    # Make a dummy environment so we can get observation and action space data
    experiment_log_dir = Path(experiment_log_dir)
    assert experiment_log_dir.exists()

    # Create dummy env to get spaces
    kwargs = {}
    config.log_dir = str(Path(config.log_dir) / (common_utils.time_stamp_str() + f"{config.suite}_{config.name}"))
    config.artefact_root = str(Path(config.log_dir) / "artefacts")
    c = copy.copy(config)
    c.monitor = False
    eg_env = Env.make(config)
    config.obs_space, config.action_space = eg_env.observation_space, eg_env.action_space
    if config.use_HER:
        kwargs["compute_reward"] = eg_env.get_reward_functor()
    config.discrete = isinstance(config.action_space, gym.spaces.Discrete)
    del (eg_env)

    # Launch the experiment
    evaluator = Runner.Evaluator(config, **kwargs)

    scores = {}
    for dir in os.listdir(experiment_log_dir / "models"):
        scores[dir] = score_reduction_fn(
            evaluator.eval_agent(
                experiment_log_dir / "models" / dir,
                episodes,
                seeds,
            )
        )
        print("Leaderboard: [")
        for i, (key, value) in enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True)):
            print(f"{i}) {value:.2f} {key}")
        print("]")


if __name__ == '__main__':
    def main():
        def env_generator():
            env_conf = Env.EnvConf()
            env_conf.suite = "classic"
            env_conf.name = "CartPole-v1"
            env_conf.render = True

            return Env.make(env_conf)

        evaluate_policy(
            r"D:\projects\ics\python_ai4ics_v2\py_ics\submodules\ICS_FastDeepQLearning\experiments\logs\2022-03-25___11-23-56classic_CartPole-v1\models\score=500.0_step=854",
            lambda : env_generator(),
            render=True,
            n_eval_episodes=2,
        )


    main()
