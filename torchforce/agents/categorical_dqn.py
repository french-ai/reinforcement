from math import floor, ceil

import torch
import torch.optim as optim
from gym.spaces import flatdim, flatten

from torchforce.agents import DQN
from torchforce.memories import ExperienceReplay
from torchforce.networks import C51Network


class CategoricalDQN(DQN):

    def __init__(self, action_space, observation_space, memory=ExperienceReplay(), neural_network=None, num_atoms=51,
                 r_min=-2, r_max=2, step_train=2, batch_size=32, gamma=0.99,
                 optimizer=None, greedy_exploration=None):

        loss = None

        super().__init__(action_space, observation_space, memory, neural_network, step_train, batch_size, gamma, loss,
                         optimizer, greedy_exploration)

        if neural_network is None:
            self.neural_network = C51Network(observation_shape=flatdim(observation_space),
                                             action_shape=flatdim(action_space))
            num_atoms = 51

        if optimizer is None:
            self.optimizer = optim.Adam(self.neural_network.parameters())

        self.num_atoms = num_atoms
        self.r_min = r_min
        self.r_max = r_max

        self.delta_z = (r_max - r_min) / float(num_atoms - 1)
        self.z = torch.Tensor([r_min + i * self.delta_z for i in range(num_atoms)])

    def get_action(self, observation):

        observation = torch.tensor([flatten(self.observation_space, observation)])

        prediction = self.neural_network.forward(observation).detach()[0]
        q_values = prediction * self.z
        q_values = torch.sum(q_values, dim=1)

        return torch.argmax(q_values).detach().item()

    def train(self):

        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

        actions = actions.int()

        predictions_next = self.neural_network.forward(next_observations).detach()
        q_values_next = predictions_next * self.z
        q_values_next = torch.sum(q_values_next, dim=2)

        actions_next = torch.argmax(q_values_next, dim=1)

        m_prob = torch.zeros((self.batch_size, self.action_space.n, self.num_atoms))

        for sample_i in range(self.batch_size):
            done = dones[sample_i]
            for j in range(self.num_atoms):
                tzj = torch.clamp(rewards[sample_i] + self.gamma * self.z[j] * (1 - done), self.r_min, self.r_max)
                bj = (tzj - self.r_min) / self.delta_z
                l, u = floor(bj), ceil(bj)

                m_prob[sample_i][actions[sample_i].item()][l] += (done + (1 - done) *
                                                                  predictions_next[sample_i][actions_next[sample_i]][
                                                                      j]) * (u - bj)
                m_prob[sample_i][actions[sample_i].item()][u] += (done + (1 - done) *
                                                                  predictions_next[sample_i][actions_next[sample_i]][
                                                                      j]) * (bj - l)

        self.optimizer.zero_grad()
        predictions = self.neural_network.forward(observations)
        loss = - predictions.log() * m_prob
        loss.sum().backward()

        self.optimizer.step()
