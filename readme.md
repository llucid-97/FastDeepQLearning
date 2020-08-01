# Fast Deep Q Learning

## Project Goals
The state of the art in Deep RL has been through [ramping up in scale](https://openai.com/blog/ai-and-compute/) scale. But with enough effort, patience and time in optimizing pipelines, people can achieve 80-90%-ish of state of art results with commodity hardware.

I'm setting out to create such from scratch to learn the intricacies of writing fast Reinforcement Learning pipelines, and combining improvements from published work to attain general algorithmic speed improvements.


I will start from simple classic control environments, then ramp up through to standard benchmarks like RoboSchool, then through to pixel-based environments like Atari.
My goal is to have a single algorithm solve all of these out-of-the-box with the same set of hyper parameters.


## The Algorithm: Asynchronous Deep Q Learning with Actor Critics
I plan on focusing on the actor critic family of Deep Q Learning ([DDPG](https://arxiv.org/abs/1509.02971)'s children), specifically [Soft Actor Critic](https://arxiv.org/abs/1812.11103) for its freedom to use both continuous and discrete policies without significant architectural changes.

The pipeline is  [APEX](https://arxiv.org/abs/1803.00933) & [SampleFactory](http://arxiv.org/abs/2006.11751) frankensteined-together:

- A **Trainer Process** trains the neural net asynchronously, grabbing data from the replay memory as fast as it can with multithreaded data loading for zero down-time
- Multiple **Environment Processes** simulate their next time steps asynchronously, then queue observations to the inference process
- An **Inference process** batches queued observations and generates actions for them as fast as it can with multithreaded data loading for zero down time, and routes the actions back to the environments

## Usage
 `main.py` configures the experiments. I haven't setup an argparse system or reading configs from file yet (on the todo list), for now, all configuration is done by edditing the config instances in main, then running it.

This was tested on windows 10 with torch 1.3.0.

## Acknowledgements

- [Pranjal Tandon's Pytorch Soft Actor Critic](https://github.com/pranz24/pytorch-soft-actor-critic) as a reference. MIT Licencse included in the module.
