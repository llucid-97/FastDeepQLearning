# Agents

The agent module is responsible for training and inference of our models.

The module provides a make() function which constructs an agent using a config

Each agent provides an act() method for inference, and is responsible for asynchronously updating its parameters.
I've chosen to do this via new processes not threads to avoid GIL contention.

Agents implemented:
- [ ] Random Network
- [ ] Soft Actor Critic [Haaronja et al](https://arxiv.org/pdf/1812.11103.pdf)