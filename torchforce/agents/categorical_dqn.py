import torch
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import flatdim, flatten

from torchforce.agents import DQN
from torchforce.memories import ExperienceReplay
from torchforce.networks import C51Network


class CategoricalDQN(DQN):

    def __init__(self, action_space, observation_space, memory=ExperienceReplay(), neural_network=None, num_atoms=51,
                 r_min=-10, r_max=10, step_train=2, batch_size=32, gamma=0.99,
                 optimizer=None, greedy_exploration=None, device=None):
        """

        :param device: torch device to run agent
        :type: torch.device
        :param action_space:
        :param observation_space:
        :param memory:
        :param neural_network:
        :param num_atoms:
        :param r_min:
        :param r_max:
        :param step_train:
        :param batch_size:
        :param gamma:
        :param optimizer:
        :param greedy_exploration:
        """
        loss = None

        super().__init__(action_space, observation_space, memory, neural_network, step_train, batch_size, gamma, loss,
                         optimizer, greedy_exploration, device=device)

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
        """ Return action choice by the agents

        :param observation: stat of environment
        :type observation: gym.Space
        """
        observation = torch.tensor([flatten(self.observation_space, observation)])

        prediction = self.neural_network.forward(observation).detach()[0]
        q_values = prediction * self.z
        q_values = torch.sum(q_values, dim=1)

        return torch.argmax(q_values).detach().item()

    def train(self):
        """

        """
        self.batch_size = 3

        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

        actions = actions.to(torch.long)
        actions = F.one_hot(actions, num_classes=self.action_space.n)

        predictions_next = self.neural_network.forward(next_observations).detach()
        q_values_next = predictions_next * self.z
        q_values_next = torch.sum(q_values_next, dim=2)

        actions_next = torch.argmax(q_values_next, dim=1)
        actions_next = actions_next.to(torch.long)
        actions_next = F.one_hot(actions_next, num_classes=self.action_space.n)

        dones = dones.view(-1, 1)

        tz = torch.clamp(rewards.view(-1, 1) + self.gamma * self.z * (1 - dones), self.r_min, self.r_max)
        b = (tz - self.r_min) / self.delta_z

        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

        m_prob = torch.zeros((self.batch_size, self.action_space.n, self.num_atoms))

        predictions_next = predictions_next[actions_next == 1, :]

        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).view(-1, 1)
        offset = offset.expand(self.batch_size, self.num_atoms)

        u_index = (u + offset).view(-1).to(torch.int64)
        l_index = (l + offset).view(-1).to(torch.int64)

        predictions_next = (dones + (1 - dones) * predictions_next)

        m_prob_action = m_prob[actions == 1, :].view(-1)
        m_prob_action.index_add_(0, u_index, (predictions_next * (u - b)).view(-1))
        m_prob_action.index_add_(0, l_index, (predictions_next * (b - l)).view(-1))

        m_prob[actions == 1, :] = m_prob_action.view(-1, self.num_atoms)

        self.optimizer.zero_grad()
        predictions = self.neural_network.forward(observations)
        loss = - predictions.log() * m_prob
        loss.sum((1, 2)).mean().backward()

        self.optimizer.step()

    def __str__(self):
        return 'CategoricalDQN-' + str(self.observation_space) + "-" + str(self.action_space) + "-" + str(
            self.neural_network) + "-" + str(self.memory) + "-" + str(self.step_train) + "-" + str(
            self.step) + "-" + str(self.batch_size) + "-" + str(self.gamma) + "-" + str(self.loss) + "-" + str(
            self.optimizer) + "-" + str(self.greedy_exploration) + "-" + str(self.num_atoms) + "-" + str(
            self.r_min) + "-" + str(self.r_max) + "-" + str(self.delta_z) + "-" + str(self.z)
