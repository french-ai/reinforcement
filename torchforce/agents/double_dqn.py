from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchforce.agents import DQN
from torchforce.memories import ExperienceReplay


class DoubleDQN(DQN):

    def __init__(self, action_space, observation_space, memory=ExperienceReplay(), neural_network=None, step_copy=500,
                 step_train=2, batch_size=32, gamma=0.99,
                 loss=None, optimizer=None, greedy_exploration=None):

        super().__init__(action_space, observation_space, memory, neural_network, step_train, batch_size, gamma, loss,
                         optimizer, greedy_exploration)

        self.neural_network_target = deepcopy(self.neural_network)
        self.copy_online_to_target()
        self.memory = memory

        self.step_copy = step_copy

        if optimizer is None:
            self.optimizer = optim.RMSprop(self.neural_network.parameters(), lr=0.00025, momentum=0.95)

    def learn(self, observation, action, reward, next_observation, done) -> None:

        super().learn(observation, action, reward, next_observation, done)

        if (self.step % self.step_copy) == 0:
            self.copy_online_to_target()

    def train(self):

        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

        actions_next = torch.argmax(self.neural_network.forward(next_observations).detach(), dim=1)
        actions_next_one_hot = F.one_hot(actions_next.to(torch.int64), num_classes=self.action_space.n)
        q_next = self.neural_network_target.forward(next_observations).detach() * actions_next_one_hot

        q = rewards + self.gamma * torch.max(q_next, dim=1)[0] * (1 - dones)

        actions_one_hot = F.one_hot(actions.to(torch.int64), num_classes=self.action_space.n)
        q_predict = torch.max(self.neural_network.forward(observations) * actions_one_hot, dim=1)[0]

        self.optimizer.zero_grad()
        loss = self.loss(q_predict, q)
        loss.backward()
        self.optimizer.step()

    def copy_online_to_target(self):
        self.neural_network_target.load_state_dict(self.neural_network.state_dict())

    def save(self, file_name, dire_name="."):
        pass

    @classmethod
    def load(cls, file_name, dire_name="."):
        pass
