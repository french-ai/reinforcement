import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Dict, Tuple

from torchforce.agents import DoubleDQN
from torchforce.explorations import Greedy, EpsilonGreedy
from torchforce.memories import ExperienceReplay
from torchforce.networks import BaseNetwork


class Network(BaseNetwork):
    def __init__(self, observation_shape=None, action_shape=None):
        super().__init__(observation_shape, action_shape)
        self.dense = nn.Linear(3, 4)

    def forward(self, x):
        x = self.dense(x)
        x = F.relu(x)
        return x


def test_double_dqn_agent_instantiation():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory=memory)
    DoubleDQN(Discrete(4), Discrete(3))


def test_double_dqn_agent_instantiation_error_action_space():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DoubleDQN("ACTION_SPACE_ERROR", Discrete(3), neural_network=network, memory=memory)


def test_double_dqn_agent_instantiation_error_observation_space():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DoubleDQN(Discrete(1), "OBSERVATION_SPACE_ERROR", neural_network=network, memory=memory)


def test_double_dqn_agent_instantiation_error_neural_network():
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DoubleDQN(Discrete(4), Discrete(3), neural_network="NEURAL_NETWORK_ERROR", memory=memory)


def test_double_dqn_agent_instantiation_error_memory():
    network = Network()

    with pytest.raises(TypeError):
        DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory="MEMORY_ERROR")


def test_double_dqn_agent_instantiation_error_loss():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory=memory, loss="LOSS_ERROR")


def test_double_dqn_agent_instantiation_error_optimizer():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory=memory, optimizer="OPTIMIZER_ERROR")


def test_double_dqn_agent_instantiation_error_greedy_exploration():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory=memory,
                  greedy_exploration="GREEDY_EXPLORATION_ERROR")


def test_double_dqn_agent_instantiation_custom_loss():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory=memory, loss=nn.MSELoss())


def test_double_dqn_agent_instantiation_custom_optimizer():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory=memory,
              optimizer=optim.RMSprop(network.parameters()))


def test_double_dqn_agent_getaction():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    agent = DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory=memory, greedy_exploration=Greedy())

    observation = [0, 1, 2]

    agent.get_action(observation)


def test_double_dqn_agent_getaction_non_greedy():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    agent = DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory=memory,
                      greedy_exploration=EpsilonGreedy(1.))

    observation = [0, 1, 2]

    agent.get_action(observation)


def test_double_dqn_agent_learn():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    agent = DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory=memory, step_copy=2)

    obs = [1, 2, 5]
    action = 0
    reward = 0
    next_obs = [5, 9, 4]
    done = False

    obs_s = [obs, obs, obs]
    actions = [1, 2, 3]
    rewards = [-2.2, 5, 4]
    next_obs_s = [next_obs, next_obs, next_obs]
    dones = [False, True, False]

    memory.extend(obs_s, actions, rewards, next_obs_s, dones)

    agent.learn(obs, action, reward, next_obs, done)
    agent.learn(obs, action, reward, next_obs, done)


def test_double_dqn_agent_episode_finished():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    agent = DoubleDQN(Discrete(4), Discrete(3), neural_network=network, memory=memory)
    agent.episode_finished()


base_list = {"box": Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32), "discrete": Discrete(3),
             "multibinary": MultiBinary(10), "multidiscrete": MultiDiscrete(10)}
dict_list = Dict(base_list)
tuple_list = Tuple(list(base_list.values()))

test_list = [*base_list.values(), dict_list, tuple_list]


def test_agent_save_load():
    for space in test_list:
        agent = DoubleDQN(observation_space=space, action_space=Discrete(2))

        agent.save(file_name="deed.pt")
        agent_l = DoubleDQN.load(file_name="deed.pt")

        assert agent.observation_space == agent_l.observation_space
        assert Discrete(2) == agent_l.action_space
        os.remove("deed.pt")

    network = Network()

    agent = DoubleDQN(observation_space=space, action_space=Discrete(2), memory=ExperienceReplay(),
                      neural_network=network, step_train=3, batch_size=12, gamma=0.50, loss=None,
                      optimizer=torch.optim.Adam(network.parameters()), step_copy=300,
                      greedy_exploration=EpsilonGreedy(0.2))

    agent.save(file_name="deed.pt")
    agent_l = DoubleDQN.load(file_name="deed.pt")

    os.remove("deed.pt")

    assert agent.observation_space == agent_l.observation_space
    assert Discrete(2) == agent_l.action_space
    assert isinstance(agent.neural_network, type(agent_l.neural_network))
    for a, b in zip(agent.neural_network.state_dict(), agent_l.neural_network.state_dict()):
        assert a == b
    assert agent.step_train == agent_l.step_train
    assert agent.batch_size == agent_l.batch_size
    assert agent.gamma == agent_l.gamma
    assert isinstance(agent.loss, type(agent_l.loss))
    for a, b in zip(agent.loss.parameters(), agent_l.loss.parameters()):
        assert a == b
    assert isinstance(agent.optimizer, type(agent_l.optimizer))
    for a, b in zip(agent.optimizer.state_dict(), agent_l.optimizer.state_dict()):
        assert a == b
    assert isinstance(agent.greedy_exploration, type(agent_l.greedy_exploration))
    assert agent.step_copy == agent_l.step_copy

    agent = DoubleDQN(observation_space=space, action_space=Discrete(2))
    agent.save(file_name="deed.pt", dire_name="./remove/")

    os.remove("./remove/deed.pt")
    os.rmdir("./remove/")

    with pytest.raises(TypeError):
        agent.save(file_name=14548)
    with pytest.raises(TypeError):
        agent.save(file_name="deed.pt", dire_name=14484)

    with pytest.raises(FileNotFoundError):
        DoubleDQN.load(file_name="deed.pt")
    with pytest.raises(FileNotFoundError):
        DoubleDQN.load(file_name="deed.pt", dire_name="/Dede/")
