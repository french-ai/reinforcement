import abc

import torch.nn as nn


class BaseNetwork(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, observation_space, action_space):
        """

        :param observation_space:
        :param action_space:
        """
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def __str__(self):
        pass
