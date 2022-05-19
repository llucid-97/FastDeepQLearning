import typing as T, copy, itertools
from queue import Queue, Full
from pathlib import Path
from franQ import Agent, Env
from franQ.common_utils import TimerTB, PyjionJit

from torch.utils.tensorboard import SummaryWriter


def env_handler(conf: T.Union[Agent.AgentConf, Env.EnvConf], idx,
                queue_put_experience: Queue, queue_get_action: Queue,
                queue_put_score: Queue = None,
                num_episodes=None, seed=None, wait_for_ranker=False,env_generator=None,preproc=False):
    """Pipeline Stage: Asynchronously handles stepping through env to get a response"""
    conf = copy.copy(conf)
    conf.instance_tag = idx
    conf.monitor = conf.monitor if isinstance(conf.monitor, bool) else conf.monitor == idx

    if env_generator is None:
        env = Env.make_mp(conf)
    else:
        env = env_generator(idx)
        if preproc:
            env = Env.make(conf).get_preprocessing_stack(conf,env)
    if seed is not None:
        env.seed(seed)

    with PyjionJit():
        logger = SummaryWriter(str(Path(conf.log_dir) / f"Runner_Env_{idx}"))
        total_step = 0
        render = conf.render if isinstance(conf.render, bool) else conf.render == idx
        episode_iterator = range(num_episodes) if num_episodes else itertools.count()
        for episode in episode_iterator:
            if episode >= conf.max_num_episodes: break

            # Reset & init all data from the environment
            score = 0
            xp = {"reward": 0.0,
                  "episode_done": False,
                  "task_done": False,
                  "idx": idx,
                  "info": {},
                  }
            xp.update(env.reset())
            for xp["episode_step"] in itertools.count():
                with TimerTB(logger, f"Pipeline_Stall{idx}", group="timers/pipeline_stats", step=total_step):
                    queue_put_experience.put(xp)
                    action, agent_state = queue_get_action.get()
                    if agent_state is not None: xp["agent_state"] = agent_state

                if xp["episode_done"]:
                    break

                with TimerTB(logger, f"Env_{idx}_Step", group="timers/runner_pipeline", step=total_step):
                    # Get new experience from environment and populate the dict
                    obs, xp["reward"], xp["episode_done"], xp["info"] = env.step(action)
                    xp.update(obs)
                    xp["task_done"] = xp["episode_done"] and not xp["info"].get('TimeLimit.truncated', False)
                    if render:
                        env.render()

                    score += xp["reward"]
                    total_step += 1

            # After Episode:
            logger.add_scalar("Env/Episode_Score", score, episode)
            logger.add_scalar("Env/TrainStep_Score", score, conf.global_step.value)
            logger.add_scalar("Env/EnvStep_Score", score, total_step)
            if queue_put_score is not None:
                try:
                    queue_put_score.put({"score":score,"step":xp["episode_step"]}, wait_for_ranker)
                except Full:
                    pass

    env.close()
    print(f"env_handler {idx} completed")
