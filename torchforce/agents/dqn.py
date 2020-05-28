from abc import ABCMeta

import torch
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Discrete, Space, flatdim, flatten

from torchforce.agents import AgentInterface
from torchforce.explorations import GreedyExplorationInterface, EpsilonGreedy
from torchforce.memories import MemoryInterface, ExperienceReplay
from torchforce.networks import SimpleNetwork


class DQN(AgentInterface, metaclass=ABCMeta):

    def __init__(self, action_space, observation_space, memory=ExperienceReplay(), neural_network=None, step_train=2,
                 batch_size=32, gamma=0.99, loss=None, optimizer=None, greedy_exploration=None):

        if not isinstance(action_space, Discrete):
            raise TypeError(
                "action_space need to be instance of gym.spaces.Space.Discrete, not :" + str(type(action_space)))
        if not isinstance(observation_space, Space):
            raise TypeError(
                "observation_space need to be instance of gym.spaces.Space.Discrete, not :" + str(
                    type(observation_space)))

        if neural_network is None and optimizer is not None:
            raise TypeError("If neural_network is None, optimizer need to be None not " + str(type(optimizer)))

        if neural_network is None:
            neural_network = SimpleNetwork(observation_shape=flatdim(observation_space),
                                           action_shape=flatdim(action_space))
        if not isinstance(neural_network, torch.nn.Module):
            raise TypeError("neural_network need to be instance of torch.nn.Module, not :" + str(type(neural_network)))

        if not isinstance(memory, MemoryInterface):
            raise TypeError(
                "memory need to be instance of torchforces.memories.MemoryInterface, not :" + str(type(memory)))

        if loss is not None and not isinstance(loss, torch.nn.Module):
            raise TypeError("loss need to be instance of torchforces.memories.MemoryInterface, not :" + str(type(loss)))

        if optimizer is not None and not isinstance(optimizer, optim.Optimizer):
            raise TypeError(
                "optimizer need to be instance of torchforces.memories.MemoryInterface, not :" + str(type(optimizer)))

        if greedy_exploration is not None and not isinstance(greedy_exploration, GreedyExplorationInterface):
            raise TypeError(
                "greedy_exploration need to be instance of torchforces.explorations.GreedyExplorationInterface, not :" + str(
                    type(greedy_exploration)))

        self.observation_space = observation_space
        self.action_space = action_space
        self.neural_network = neural_network
        self.memory = memory

        self.step_train = step_train
        self.step = 0
        self.batch_size = batch_size

        self.gamma = gamma

        if loss is None:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = loss

        if optimizer is None:
            self.optimizer = optim.SGD(self.neural_network.parameters(), lr=0.01)
        else:
            self.optimizer = optimizer

        if greedy_exploration is None:
            self.greedy_exploration = EpsilonGreedy(0.1)
        else:
            self.greedy_exploration = greedy_exploration

    def get_action(self, observation):

        if not self.greedy_exploration.be_greedy(self.step):
            return self.action_space.sample()

        observation = torch.tensor([flatten(self.observation_space, observation)])

        q_values = self.neural_network.forward(observation)

        return torch.argmax(q_values).detach().item()

    def learn(self, observation, action, reward, next_observation, done) -> None:
        
        self.memory.append(observation, action, reward, next_observation, done)
        self.step += 1

        if (self.step % self.step_train) == 0:
            self.train()

    def episode_finished(self) -> None:
        pass

    def train(self):

        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

        q = rewards + self.gamma * torch.max(self.neural_network.forward(next_observations), dim=1)[0].detach() * (
                1 - dones)

        actions_one_hot = F.one_hot(actions.to(torch.int64), num_classes=self.action_space.n)
        q_values_predict = self.neural_network.forward(observations) * actions_one_hot
        q_predict = torch.max(q_values_predict, dim=1)

        self.optimizer.zero_grad()
        loss = self.loss(q_predict[0], q)
        loss.backward()
        self.optimizer.step()

    def save(self, file_name, dire_name="."):
        pass

    @classmethod
    def load(cls, file_name, dire_name="."):
        pass
