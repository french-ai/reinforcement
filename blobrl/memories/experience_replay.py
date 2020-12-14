import numpy as np
import torch
from collections import deque

from blobrl.memories import MemoryInterface


class ExperienceReplay(MemoryInterface):

    def __init__(self, max_size=5000):
        """

        :param max_size:
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def append(self, observation, action, reward, next_observation, done):
        """

        :param observation:
        :param action:
        :param reward:
        :param next_observation:
        :param done:
        """
        self.buffer.append([observation, action, reward, next_observation, done])

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
        idxs = np.random.randint(len(self.buffer), size=batch_size)

        batch = np.array([self.buffer[idx] for idx in idxs])

        return [torch.Tensor(list(V)).to(device=device) for V in batch.T]

    def __str__(self):
        return 'ExperienceReplay-' + str(self.max_size)
