import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Discrete

from torchforce.agents import DQN
from torchforce.explorations import Greedy, EpsilonGreedy
from torchforce.memories import ExperienceReplay


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.dense = nn.Linear(4, 4)

    def forward(self, x):
        x = self.dense(x)
        x = F.relu(x)
        return x


def test_dqn_agent_instantiation():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    DQN(Discrete(4), Discrete(4), memory, neural_network=network)


def test_dqn_agent_instantiation_error_action_space():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DQN(None, Discrete(1), memory, neural_network=network)


def test_dqn_agent_instantiation_error_observation_space():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DQN(Discrete(1), None, memory, neural_network=network)


def test_dqn_agent_instantiation_error_neural_network():
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DQN(Discrete(4), Discrete(4), memory, neural_network=154)

    DQN(Discrete(4), Discrete(4), memory, neural_network=None)


def test_dqn_agent_instantiation_error_memory():
    network = Network()

    with pytest.raises(TypeError):
        DQN(Discrete(4), Discrete(4), None, neural_network=network)


def test_dqn_agent_instantiation_error_loss():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DQN(Discrete(4), Discrete(4), memory, neural_network=network, loss="LOSS_ERROR")


def test_dqn_agent_instantiation_error_optimizer():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DQN(Discrete(4), Discrete(4), memory, neural_network=network, optimizer="OPTIMIZER_ERROR")


def test_dqn_agent_instantiation_error_greedy_exploration():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    with pytest.raises(TypeError):
        DQN(Discrete(4), Discrete(4), memory, neural_network=network, greedy_exploration="GREEDY_EXPLORATION_ERROR")


def test_dqn_agent_instantiation_custom_loss():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    DQN(Discrete(4), Discrete(4), memory, neural_network=network, loss=nn.MSELoss())


def test_dqn_agent_instantiation_custom_optimizer():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    DQN(Discrete(4), Discrete(4), memory, neural_network=network, optimizer=optim.RMSprop(network.parameters()))

    with pytest.raises(TypeError):
        DQN(Discrete(4), Discrete(4), memory, neural_network=None, optimizer=optim.RMSprop(network.parameters()))


def test_dqn_agent_getaction():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    agent = DQN(Discrete(4), Discrete(4), memory, neural_network=network, greedy_exploration=Greedy())

    observation = [0, 1, 2]

    agent.get_action(observation)


def test_dqn_agent_getaction_non_greedy():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    agent = DQN(Discrete(4), Discrete(4), memory, neural_network=network, greedy_exploration=EpsilonGreedy(1.))

    observation = [0, 1, 2]

    agent.get_action(observation)


def test_dqn_agent_learn():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    agent = DQN(Discrete(4), Discrete(4), memory, neural_network=network)

    obs = [1, 2, 5, 0]
    action = 0
    reward = 0
    next_obs = [5, 9, 4, 0]
    done = False

    obs_s = [obs, obs, obs]
    actions = [1, 2, 3]
    rewards = [-2.2, 5, 4]
    next_obs_s = [next_obs, next_obs, next_obs]
    dones = [False, True, False]

    memory.extend(obs_s, actions, rewards, next_obs_s, dones)

    agent.learn(obs, action, reward, next_obs, done)
    agent.learn(obs, action, reward, next_obs, done)


def test_dqn_agent_episode_finished():
    network = Network()
    memory = ExperienceReplay(max_size=5)

    agent = DQN(Discrete(4), Discrete(4), memory, neural_network=network)
    agent.episode_finished()
