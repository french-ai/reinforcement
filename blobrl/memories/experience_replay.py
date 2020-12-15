import numpy as np
import torch
from collections import deque

from blobrl.memories import MemoryInterface


class ExperienceReplay(MemoryInterface):

    def __init__(self, max_size=5000, gamma=0.0):
        """
        Create ExperienceReplay with buffersize equal to max_size

        :param max_size: size max of buffer
        :type max_size: int
        :param gamma: gamma from Temporal Distance objective
        :type gamma: float [0,1]
        """
        self.buffer = deque(maxlen=max_size)
        if not 0 <= gamma <= 1:
            raise ValueError("gamma need to be in range [0,1] not " + str(gamma))
        self.gamma = gamma

    def append(self, observation, action, reward, next_observation, done):
        """
        Store one couple of value

        :param observation:
        :param action:
        :param reward:
        :param next_observation:
        :param done:
        """
        self.buffer.append([observation, action, reward, next_observation, done])

    def extend(self, observations, actions, rewards, next_observations, dones):
        """
        Store many couple of value

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
        returns *batch_size* of samples

        :param device: torch device to run agent
        :type device: torch.device
        :param batch_size:
        :type batch_size: int
        :return: list<Tensor>
        """
        idxs = np.random.randint(len(self.buffer), size=batch_size)

        batch = np.array([self.get_sample(idx) for idx in idxs])

        return [torch.Tensor(list(V)).to(device=device) for V in batch.T]

    def get_sample(self, idx):
        """
        returns sample at idx position

        :param idx: torch device to run agent
        :type idx: int
        :return: [observation, action, reward, next_observation, done]
        """
        return self.buffer[idx]

    def __str__(self):
        return 'ExperienceReplay-' + str(self.buffer.maxlen) + '-' + str(self.gamma)
