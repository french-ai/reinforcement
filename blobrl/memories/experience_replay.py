import numpy as np
import torch

from blobrl.memories import MemoryInterface


class ExperienceReplay(MemoryInterface):

    def __init__(self, max_size=5000):
        """

        :param max_size:
        """
        self.max_size = max_size
        self.buffer = np.empty(shape=(self.max_size, 5), dtype=np.object)
        self.index = 0
        self.size = 0

    def append(self, observation, action, reward, next_observation, done):
        """

        :param observation:
        :param action:
        :param reward:
        :param next_observation:
        :param done:
        """
        self.buffer[self.index] = np.array([np.array(observation), action, reward, np.array(next_observation), done])
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def extend(self, observations, actions, rewards, next_observations, dones):
        """

        :param observations:
        :param actions:
        :param rewards:
        :param next_observations:
        :param dones:
        """
        for o, a, r, n, d in zip(observations, actions, rewards, next_observations, dones):
            self.append(o, a, r, n, d)

    def sample(self, batch_size, device):
        """

        :param device:
        :param batch_size:
        :return:
        """
        idxs = np.random.randint(self.size, size=batch_size)

        return [torch.Tensor(list(V)).to(device=device) for V in self.buffer[idxs].T]

    def __str__(self):
        return 'ExperienceReplay-' + str(self.max_size)
