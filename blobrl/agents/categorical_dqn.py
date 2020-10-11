import torch
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Discrete, Space, flatten

from blobrl.agents import DQN
from blobrl.memories import ExperienceReplay
from blobrl.networks import C51Network


class CategoricalDQN(DQN):

    def __init__(self, observation_space, action_space, memory=ExperienceReplay(), network=None, num_atoms=51,
                 r_min=-10, r_max=10, step_train=2, batch_size=32, gamma=0.99,
                 optimizer=None, greedy_exploration=None, device=None):
        """

        :param device: torch device to run agent
        :type: torch.device
        :param action_space:
        :param observation_space:
        :param memory:
        :param network:
        :param num_atoms:
        :param r_min:
        :param r_max:
        :param step_train:
        :param batch_size:
        :param gamma:
        :param optimizer:
        :param greedy_exploration:
        """
        if network is None and optimizer is None:
            network = C51Network(observation_space=observation_space,
                                 action_space=action_space)
            num_atoms = 51

            optimizer = optim.Adam(network.parameters())

        super().__init__(observation_space=observation_space, action_space=action_space, memory=memory,
                         network=network, step_train=step_train, batch_size=batch_size, gamma=gamma,
                         loss=None, optimizer=optimizer, greedy_exploration=greedy_exploration, device=device)

        self.num_atoms = num_atoms
        self.r_min = r_min
        self.r_max = r_max

        self.delta_z = (r_max - r_min) / float(num_atoms - 1)
        self.z = torch.tensor([r_min + i * self.delta_z for i in range(num_atoms)], device=self.device)

    def get_action(self, observation):
        """ Return action choice by the agents

        :param observation: stat of environment
        :type observation: gym.Space
        """
        if not self.greedy_exploration.be_greedy(self.step) and self.with_exploration:
            return self.action_space.sample()

        observation = torch.tensor([flatten(self.observation_space, observation)], device=self.device).float()

        prediction = self.network.forward(observation)

        def return_values(values):
            if isinstance(values, list):
                return [return_values(v) for v in values]
            else:
                q_values = values * self.z
                q_values = torch.sum(q_values, dim=1)
                return torch.argmax(q_values).detach().item()

        return return_values(prediction)

    def apply_loss(self, next_prediction, prediction, actions, rewards, next_observations, dones, len_space):
        if isinstance(next_prediction, list):
            [self.apply_loss(n, p, a, rewards, next_observations, dones, c) for n, p, a, c in
             zip(next_prediction, prediction, actions.permute(1, 0, *[i for i in range(2, len(actions.shape))]),
                 len_space)]
        else:
            actions = F.one_hot(actions.long(), num_classes=len_space)

            q_values_next = next_prediction * self.z
            q_values_next = torch.sum(q_values_next, dim=2)

            actions_next = torch.argmax(q_values_next, dim=1).long()
            actions_next = F.one_hot(actions_next.long(), num_classes=len_space)

            dones = dones.view(-1, 1)

            tz = torch.clamp(rewards.view(-1, 1) + self.gamma * self.z * (1 - dones), self.r_min, self.r_max)
            b = (tz - self.r_min) / self.delta_z

            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            m_prob = torch.zeros((self.batch_size, len_space, self.num_atoms), device=self.device)

            predictions_next = next_prediction[actions_next == 1, :]

            offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size,
                                    device=self.device).view(-1,
                                                             1)
            offset = offset.expand(self.batch_size, self.num_atoms)

            u_index = (u + offset).view(-1).to(torch.int64)
            l_index = (l + offset).view(-1).to(torch.int64)

            predictions_next = (dones + (1 - dones) * predictions_next)

            m_prob_action = m_prob[actions == 1, :].view(-1)
            m_prob_action.index_add_(0, u_index, (predictions_next * (u - b)).view(-1))
            m_prob_action.index_add_(0, l_index, (predictions_next * (b - l)).view(-1))

            m_prob[actions == 1, :] = m_prob_action.view(-1, self.num_atoms)

            self.optimizer.zero_grad()

            loss = - prediction.log() * m_prob
            loss.sum((1, 2)).mean().backward(retain_graph=True)

    def __str__(self):
        return 'CategoricalDQN-' + str(self.observation_space) + "-" + str(self.action_space) + "-" + str(
            self.network) + "-" + str(self.memory) + "-" + str(self.step_train) + "-" + str(
            self.step) + "-" + str(self.batch_size) + "-" + str(self.gamma) + "-" + str(self.loss) + "-" + str(
            self.optimizer) + "-" + str(self.greedy_exploration) + "-" + str(self.num_atoms) + "-" + str(
            self.r_min) + "-" + str(self.r_max) + "-" + str(self.delta_z) + "-" + str(self.z)
