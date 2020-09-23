import numpy as np
from torch import nn
from gym.spaces import Space, flatdim
from .utils import get_last_layers
from blobrl.networks import BaseDuelingNetwork


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

        super().__init__(observation_space=observation_space, action_space=action_space)

        self.features = nn.Sequential()
        self.features.add_module("NetWorkSimple_Linear_Input", nn.Linear(np.prod(flatdim(self.observation_space)), 64))
        self.features.add_module("NetWorkSimple_LeakyReLU_Input", nn.LeakyReLU())
        self.features.add_module("NetWorkSimple_Linear_1", nn.Linear(64, 64))
        self.features.add_module("NetWorkSimple_LeakyReLU_1", nn.LeakyReLU())
        self.features.add_module("NetWorkSimple_Linear_Output", nn.Linear(64, 64))

        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            get_last_layers(self.action_space, last_dim=64)
        )

        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def __str__(self):
        return 'SimpleDuelingNetwork-' + str(self.observation_space) + "-" + str(self.action_space)
