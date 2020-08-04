# Fast Deep Q Learning

<img align="right" width="200" height="183" src="https://i.redd.it/gfkg9wxfmww11.png">

Combining improvements in deep Q Learning for fast and stable training with a modular, configurable agent.
[Pranjal Tandon's Pytorch Soft Actor Critic](https://github.com/pranz24/pytorch-soft-actor-critic) is used as a baseline. I've added the following optional components atop it:

 - Asynchronous Environment rollouts and Parameter Updates base on a combination of [Horgan et al's APEX Pipeline](https://arxiv.org/abs/1803.00933) and [Petrenko et al's SampleFactory](https://arxiv.org/abs/2006.11751). [Discussed here](https://medium.com/@hexxagon6/writing-fast-deep-q-learning-pipelines-on-commodity-hardware-a3c59cdda429)
 - [He et al's variant of n-step returns](https://arxiv.org/abs/1611.01606) : using the sampled return as a lower-bound constraint (penalty actually) on Q predictions to accelerate convergence

WIP:
 - A Discrete Policy for SAC based on [Wah Loon Keng's SLM-Lab baseline](https://github.com/kengz/SLM-Lab)
 - State dependent exploration based on [Raffin & Stulp's gSDE](https://arxiv.org/abs/2005.05719) to make SAC more robust to environments that act like low-pass filters

## Motivation:
The state of the art in Deep RL has been through [ramping up in scale](https://openai.com/blog/ai-and-compute/) scale. But with enough effort, patience and time in optimizing pipelines, people can achieve 80-90%-ish of state of art results with commodity hardware.

I'm setting out to create such from scratch to learn the intricacies of writing fast Reinforcement Learning pipelines, and combining improvements from published work to attain general algorithmic speed improvements.


I will start from simple classic control environments, then ramp up through to standard benchmarks like RoboSchool, then through to pixel-based environments like Atari.
My goal is to have a single algorithm solve all of these out-of-the-box with the same set of hyper parameters.


## Usage
 `main.py` configures the experiments. I haven't setup an argparse system or reading configs from file yet (on the todo list), for now, all configuration is done by edditing the config instances in main, then running it.

This was tested on windows 10 with torch 1.3.0.
