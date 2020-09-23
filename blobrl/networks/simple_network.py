import numpy as np
import torch.nn as nn
from gym.spaces import Space, flatdim
from .utils import get_last_layers
from blobrl.networks import BaseNetwork


class SimpleNetwork(BaseNetwork):
    def __init__(self, observation_space, action_space):
        """

        :param observation_space:
        :param action_space:
        """
        super().__init__(observation_space=observation_space, action_space=action_space)

        if not isinstance(observation_space, Space):
            raise TypeError("observation_space need to be Space not " + str(type(observation_space)))
        if not isinstance(action_space, Space):
            raise TypeError("action_space need to be Space not " + str(type(action_space)))

        self.network = nn.Sequential()
        self.network.add_module("NetWorkSimple_Linear_Input", nn.Linear(np.prod(flatdim(self.observation_space)), 64))
        self.network.add_module("NetWorkSimple_LeakyReLU_Input", nn.LeakyReLU())
        self.network.add_module("NetWorkSimple_Linear_1", nn.Linear(64, 64))
        self.network.add_module("NetWorkSimple_LeakyReLU_1", nn.LeakyReLU())
        self.network.add_module("NetWorkSimple_Linear_Output", get_last_layers(self.action_space, last_dim=64))

    def forward(self, observation):
        """

        :param observation:
        :return:
        """
        x = observation.view(observation.shape[0], -1)
        return self.network(x)

    def __str__(self):
        return 'SimpleNetwork-' + str(self.observation_space) + "-" + str(self.action_space)
