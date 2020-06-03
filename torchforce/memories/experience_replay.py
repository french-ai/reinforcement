import numpy as np
import torch

from torchforce.memories import MemoryInterface


class ExperienceReplay(MemoryInterface):

    def __init__(self, max_size=300):
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
        datas = np.array((observations, actions, rewards, next_observations, dones)).T

        len_datas = len(datas)

        if len_datas > self.max_size:
            datas = datas[-self.max_size:]
            len_datas = self.max_size

        datas[:, 0] = [np.array(observation) for observation in datas[:, 0]]
        datas[:, 3] = [np.array(next_observation) for next_observation in datas[:, 3]]

        idx_max = self.index + len_datas

        if idx_max > self.max_size:
            idx_max = self.max_size - idx_max
            self.buffer[self.index:self.max_size] = datas[:idx_max]
            self.buffer[:idx_max] = datas[idx_max:]
        else:
            self.buffer[self.index:idx_max] = datas

        self.size = min(self.size + len_datas, self.max_size)
        self.index = idx_max

    def sample(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        idxs = np.random.randint(self.size, size=batch_size)

        return [torch.Tensor(list(V)) for V in self.buffer[idxs].T]

    def __str__(self):
        return 'ExperienceReplay-' + str(self.max_size)
