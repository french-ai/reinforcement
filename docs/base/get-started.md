Getting started
===============

# Install BlobRL
 Follow [installation](./installation.md).

### Initializing an environment
```python
import gym
env = gym.make("CartPole-v1")
```

### Initializing an agent

```python
from blobrl.agents import AgentRandom
action_space = env.action_space
observation_space = env.observation_space
agent = AgentRandom(observation_space=observation_space, action_space=action_space)
```

### Training

Create Trainer
```python
from blobrl import Trainer
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