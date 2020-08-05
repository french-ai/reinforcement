# Results

All results is show in this directory.

There are one subdirectory by environment used.

## CartPole

Example with agent: XXX, network: XXX,  Algo: Adam, Memories: ExperienceReplay, Step train: XXX, Batch size: XXX, Gamma: XXX, Exploration: XXX, Learning rate: XXX

![CartPoleExemple.gif](https://github.com/french-ai/reinforcement/blob/master/result/ressources/cartpole.gif)


![CartPoleTrainning.png](https://github.com/french-ai/reinforcement/blob/master/result/ressources/CartPoleTrainning.gif)
![CartPoleEvaluation.png](https://github.com/french-ai/reinforcement/blob/master/result/ressources/CartPoleEvaluation.gif)

### Parameters range

We test to train all this agent with this parameters

* Agent
  * Algo : [DQN, DoubleDQN, DuelingDQN, CategoricalDQN]

  * Step train : [1, 4, 32]
  
  * Batch size : [32, 64]
  
  * Gamma : [0.99]
  
  * Exploration : [EpsilonGreedy(0.1),
                         EpsilonGreedy(0.6),
                         AdaptativeEpsilonGreedy(0.3, 0.1, 50000, 0),
                         AdaptativeEpsilonGreedy(0.8, 0.2, 50000, 0)]
* Network

For _DQN, DoubleDQN_ : SimpleNetwork

For _DuelingDQN_ : SimpleDuelingNetwork

For _CategoricalDQN_ : C51Network
* Optimizer
  * Algo : Adam
  * Learning rate : [0.1, 0.001, 0.001]
* Memories
  * Algo : [ExperienceReplay]

### Result analysis

#### Agent performance


#### Parameters importance


### Reproduce this result

```batch
python result.py --env "CartPole-v1" --max_episode 300
```

## Env2
*coming soon*