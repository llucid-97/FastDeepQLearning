import typing as T, copy, os
from pathlib import Path

import gym, numpy as np

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
