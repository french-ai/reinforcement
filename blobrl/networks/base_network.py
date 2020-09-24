import abc
from gym.spaces import Space
import torch.nn as nn


class BaseNetwork(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, observation_space, action_space):
        """

        :param observation_space:
        :param action_space:
        """
        super().__init__()

        if not isinstance(observation_space, Space):
            raise TypeError("observation_space need to be Space not " + str(type(observation_space)))
        if not isinstance(action_space, Space):
            raise TypeError("action_space need to be Space not " + str(type(action_space)))

        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def forward(self, observation):
        """

        :param observation:
        :return:
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
