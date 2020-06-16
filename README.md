TorchForce : Reinforcement Learning library with Pytorch
============

[![Read the Docs](https://img.shields.io/readthedocs/torchforce?style=for-the-badge)](https://torchforce.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://img.shields.io/travis/french-ai/reinforcement?branch=master.svg&style=for-the-badge)](https://travis-ci.org/french-ai/reinforcement)
[![CodeFactor](https://www.codefactor.io/repository/github/french-ai/reinforcement/badge?style=for-the-badge)](https://www.codefactor.io/repository/github/french-ai/reinforcement)
[![Codecov](https://img.shields.io/codecov/c/github/french-ai/reinforcement?style=for-the-badge)](https://codecov.io/gh/french-ai/reinforcement)
[![Discord](https://img.shields.io/badge/discord-chat-7289DA.svg?logo=Discord&style=for-the-badge)](https://discord.gg/f5MZP2K)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](https://github.com/french-ai/reinforcement/blob/master/LICENSE)

We want to create library for reinforcement learning with *pytorch*. 

## Installation

## Pytorch

For installing *pytorch* follow [Quick Start Locally](https://pytorch.org/) for your config.

## Torchforce
Download files:

```bash
git clone https://github.com/french-ai/reinforcement.git
```

Move to reinforcement directory:

```bash
cd reinforcement
```
Install torchforce

- to use it:

```bash
pip install .
```

- to help development:

```bash
pip install ".[dev]" .
```

### Get Started
### Initializing an environment
```python
import gym
env = gym.make("CartPole-v1")
```

### Initializing an agent

```python
from torchforce.agents import AgentRandom
action_space = env.action_space
observation_space = env.observation_space
agent = AgentRandom(observation_space=observation_space, action_space=action_space)
```

### Training

Create Trainer
```python
from torchforce import Trainer
trainer = Trainer(environment=env, agent=agent)
```
Start training:
```python
trainer.train(render=True)
```
Visualize training metrics:
```bash
tensorboard --logdir runs
```

### Evaluation
*Not implemented yet*

## Environments

We will use GYM environments for moments.

*No environments yet*

Watch [TODO](./TODO.md#environments-list) for environments in coming.

## Agents
*No agents yet*

## Examples
*No Examples yet*

## Results
You can see more [results](./results/README.md)

*No result yet*
