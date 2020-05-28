from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Discrete, Space, flatdim, flatten

from torchforce.agents import DQN
from torchforce.explorations import GreedyExplorationInterface, EpsilonGreedy
from torchforce.memories import MemoryInterface, ExperienceReplay
from torchforce.networks import C51Network


class DistributionalDQN(DQN):

    def __init__(self, action_space, observation_space, memory=ExperienceReplay(), neural_network = None, num_atoms=51, r_min=-2, r_max=2, step_train=2, batch_size=32, gamma=0.99,
                 loss=None, optimizer=None, greedy_exploration=None):

        super().__init__(action_space, observation_space, memory, neural_network, step_train, batch_size, gamma, loss, optimizer, greedy_exploration)
               
        if neural_network is None:
            self.neural_network = C51Network(observation_shape=flatdim(observation_space),
                                           action_shape=flatdim(action_space))
        if loss is None:
            self.loss = torch.nn.CrossEntropyLoss()

        if optimizer is None:
            self.optimizer = optim.Adam(self.neural_network.parameters(), lr=0.01)

        self.num_atoms = num_atoms
        self.r_min = r_min
        self.r_max = r_max

        self.delta_support = (r_max - r_min) / float(num_atoms - 1)
        self.support = torch.Tensor([r_min + i * self.delta_support for i in range(num_atoms)])

    def get_action(self, observation):

        observation = torch.tensor([flatten(self.observation_space, observation)])

        prediction = self.neural_network.forward(observation).detach()[0]
        q_values = prediction * self.support
        q_values = torch.sum(q_values, dim=1)

        return torch.argmax(q_values).detach().item()

    def learn(self, observation, action, reward, next_observation, done) -> None:
        pass

    def train(self):        
        pass