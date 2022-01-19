import typing as T, copy, itertools
from queue import Queue, Full
from pathlib import Path
from franQ import Agent, Env
from franQ.common_utils import TimerTB

from torch.utils.tensorboard import SummaryWriter

def env_handler(conf: T.Union[Agent.AgentConf, Env.EnvConf], idx,
                queue_put_experience: Queue, queue_get_action: Queue,
                queue_put_score: Queue = None,
                num_episodes=None, seed=None, wait_for_ranker=False):
    """Pipeline Stage: Asynchronously handles stepping through env to get a response"""
    conf = copy.copy(conf)
    conf.instance_tag = idx
    conf.monitor = conf.monitor if isinstance(conf.monitor, bool) else conf.monitor == idx

    env = Env.make_mp(conf)
    if seed is not None:
        env.seed(seed)

    logger = SummaryWriter(str(Path(conf.log_dir) / f"Runner_Env_{idx}"))
    total_step = 0
    render = conf.render if isinstance(conf.render, bool) else conf.render == idx

    episode_iterator = range(num_episodes) if num_episodes else itertools.count()
    for episode in episode_iterator:
        if episode >= conf.max_num_episodes: break

        # Reset & init all data from the environment
        score = 0
        experience = {"reward": 0.0,
                      "episode_done": False,
                      "task_done": False,
                      "idx": idx,
                      "info": {}
                      }
        experience.update(env.reset())
        for experience["episode_step"] in itertools.count():
            with TimerTB(logger, f"Pipeline_Stall{idx}", group="timers/pipeline_stats",
                         step=total_step):
                # Get action form agent
                queue_put_experience.put(experience)
                action = queue_get_action.get()

            if experience["episode_done"]:
                break

            with TimerTB(logger, f"Env_{idx}_Step", group="timers/runner_pipeline",
                         step=total_step):
                # Get new experience from environment and populate the dict
                obs, experience["reward"], experience["episode_done"], experience["info"] = env.step(action)
                experience.update(obs)
                experience["task_done"] = experience["episode_done"] and not experience["info"].get(
                    'TimeLimit.truncated', False)
                if render:
                    env.render()

                score += experience["reward"]
                total_step += 1

        # After Episode:
        logger.add_scalar("Env/Episode_Score", score, episode)
        logger.add_scalar("Env/TrainStep_Score", score, conf.global_step.value)
        # logger.add_scalar("Env/EnvStep_Score", score, total_step)
        if queue_put_score is not None:
            try:
                queue_put_score.put(score, wait_for_ranker)
            except Full:
                pass

    env.close()
