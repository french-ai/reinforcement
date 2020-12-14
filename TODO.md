

# ToDo list

- [x] Update [requirements.txt](./requirements.txt)
- [x] Design the architecture of code
- [x] Choice test tool and init them
- [x] Choice docs tool and init this 
- [x] Config codecov
- [x] Config codefactor
- [x] Create Code style standard
- [ ] Document it in [CONTRIBUTING.md](./CONTRIBUTING.md)
- [x] List Agents for starting project
- [x] List Environments for start project
- [x] Add gpu option
- [x] Render on notebook/collab

# Agents list

- [x] Random Agent
- [x] Constant Agent


- [x] Deep Q Network (Mnih *et al.*, [2013](https://arxiv.org/abs/1312.5602))
- [ ] Deep Recurrent Q Network (Hausknecht *et al.*, [2015](https://arxiv.org/abs/1507.06527))
- [ ] Persistent Advantage Learning (Bellamare *et al.*, [2015](https://arxiv.org/abs/1512.04860))
- [x] Double Deep Q Network (van Hasselt *et al.*, [2016](https://arxiv.org/abs/1509.06461))
- [x] Dueling Q Network (Wang *et al.*, [2016](https://arxiv.org/abs/1511.06581))
- [ ] Bootstraped Deep Q Network (Osband *et al.*, [2016](https://arxiv.org/abs/1602.04621))
- [ ] Continuous Deep Q Network (Gu*et al.*, [2016](https://arxiv.org/abs/1603.00748))
- [x] Categorical Deep Q Network (Bellamare *et al.*, [2017](https://arxiv.org/abs/1707.06887))
- [ ] Quantile Regression DQN (Dabney et al, [2017](https://arxiv.org/abs/1710.10044))


- [ ] Rainbow (Hessel *et al.*, [2017](https://arxiv.org/abs/1710.02298))
- [ ] Quantile Regression Deep Q Network (Dabney *et al.*, [2017](https://arxiv.org/abs/1710.10044))


- [ ] Soft Actor-Critic (Haarnoja et al, [2018](https://arxiv.org/abs/1801.01290))


- [ ] Vanilla Policy Gradient ([2000](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf))


- [ ] Deep Deterministic Policy Gradient (Lillicrap et al, [2015](https://arxiv.org/abs/1509.02971))
- [ ] Twin Delayed DDPG (Fujimoto et al, [2018](https://arxiv.org/abs/1802.09477))


- [ ] Trust Region Policy Optimization (Schulman *et al.*, [2015](https://arxiv.org/abs/1502.05477))
- [ ] Proximal Policy Optimizations (Schulman *et al.*, [2017](https://arxiv.org/abs/1707.06347))


- [ ] A2C (Mnih et al, [2016](https://arxiv.org/abs/1602.01783))
- [ ] A3C (Mnih et al, [2016](https://arxiv.org/abs/1602.01783))


- [ ] Hindsight Experience Replay (Andrychowicz et al, [2017](https://arxiv.org/abs/1707.01495))

# Network

- [x] base network support discrete action space
- [x] base network support continuous action space
- [x] base network support discrete observation space
- [x] base network support continuous observation space
- [x] simple network support discrete/continuous action/observation space
- [x] c51 network support discrete action/observation space
- [x] base dueling network support discrete/continuous action/observation space
- [x] simple dueling network support discrete/continuous action/observation space

# Explorations list

- [x] Random
- [x] Epsilon Greedy
- [ ] Intrinsic Curiosity Module (Pathak *et al.*, [2017](https://arxiv.org/abs/1705.05363))
- [ ] Random Network Distillation (Burda *et al.*, [2017](https://arxiv.org/abs/1810.12894))

# Memories list

- [ ] No memory (= model based)
- [ ] Trajectory replay
- [x] Experience Replay (Lin, [1992](https://link.springer.com/article/10.1007/BF00992699))
- [ ] Prioritized Experience Replay (Schaul *et al.*, [2015](https://arxiv.org/abs/1511.05952))
- [ ] Hindsight Experience Replay (Andrychowicz *et al.*, [2017](https://arxiv.org/abs/1707.01495))

- [ ] Add temporal difference option in all memories

# Environments list

- [x] Gym CartPole
