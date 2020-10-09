import numpy as np
from torch import nn
from gym.spaces import Space, flatdim
from .utils import get_last_layers
from blobrl.networks import BaseDuelingNetwork, SimpleNetwork


class SimpleDuelingNetwork(BaseDuelingNetwork):
    def __init__(self, observation_space, action_space):
        """

        :param observation_space:
        :param action_space:
        """

        if not isinstance(observation_space, Space):
            raise TypeError("observation_space need to be Space not " + str(type(observation_space)))
        if not isinstance(action_space, Space):
            raise TypeError("action_space need to be Space not " + str(type(action_space)))

        super().__init__(network=SimpleNetwork(observation_space=observation_space, action_space=action_space))

        self.value_outputs = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def __str__(self):
        return 'SimpleDuelingNetwork-' + str(self.observation_space) + "-" + str(self.action_space)
