from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Discrete

from torchforce.agents import AgentInterface
from torchforce.explorations import GreedyExplorationInterface, EpsilonGreedy
from torchforce.memories import MemoryInterface


class DoubleDQN(AgentInterface):

    def __init__(self, action_space, neural_network, memory, step_copy=1000, step_train=2, batch_size=8, gamma=0.99,
                 loss=None, optimizer=None, greedy_exploration=None):

        if not isinstance(action_space, Discrete):
            raise TypeError(
                "action_space need to be instance of gym.spaces.Space.Discrete, not :" + str(type(action_space)))

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

        self.action_space = action_space
        self.neural_network_online = neural_network
        self.neural_network_target = deepcopy(neural_network)
        self.copy_online_to_target()
        self.memory = memory

        self.step_copy = step_copy
        self.step_train = step_train
        self.step = 0
        self.batch_size = batch_size

        self.gamma = gamma

        if loss is None:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = loss

        if optimizer is None:
            self.optimizer = optim.RMSprop(self.neural_network_online.parameters())
        else:
            self.optimizer = optimizer

        if greedy_exploration is None:
            self.greedy_exploration = EpsilonGreedy(0.1)
        else:
            self.greedy_exploration = greedy_exploration

    def get_action(self, observation):

        if not self.greedy_exploration.be_greedy(self.step):
            return self.action_space.sample()

        observation = torch.tensor(observation)
        observation = observation.view(1, -1)

        return torch.argmax(self.neural_network_online.forward(observation))

    def learn(self, observation, action, reward, next_observation, done) -> None:

        self.memory.append(observation, action, reward, next_observation, done)
        self.step += 1

        if (self.step % self.step_train) == 0:
            self.train()

        if (self.step % self.step_copy) == 0:
            self.copy_online_to_target()

    def episode_finished(self) -> None:
        pass

    def train(self):

        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

        actions_next = torch.argmax(self.neural_network_online.forward(next_observations).detach(), dim=1)
        actions_next_one_hot = F.one_hot(actions_next.to(torch.int64), num_classes=self.action_space.n)
        q_next = self.neural_network_target.forward(next_observations).detach() * actions_next_one_hot

        q = rewards + self.gamma * torch.max(q_next, dim=1)[0] * (1 - dones)

        actions_one_hot = F.one_hot(actions.to(torch.int64), num_classes=self.action_space.n)
        q_predict = torch.max(self.neural_network_online.forward(observations) * actions_one_hot, dim=1)[0]

        self.optimizer.zero_grad()
        loss = self.loss(q_predict, q)
        loss.backward()
        self.optimizer.step()

    def copy_online_to_target(self):
        self.neural_network_target.load_state_dict(self.neural_network_online.state_dict())
