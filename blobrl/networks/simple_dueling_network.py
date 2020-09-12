import numpy as np
from torch import nn

from blobrl.networks import BaseDuelingNetwork


class SimpleDuelingNetwork(BaseDuelingNetwork):
    def __init__(self, observation_shape, action_shape):
        """

        :param observation_shape:
        :param action_shape:
        """
        super().__init__(observation_shape=observation_shape, action_shape=action_shape)

        self.features = nn.Sequential()
        self.features.add_module("NetWorkSimple_Linear_Input", nn.Linear(np.prod(self.observation_space), 64))
        self.features.add_module("NetWorkSimple_LeakyReLU_Input", nn.LeakyReLU())
        self.features.add_module("NetWorkSimple_Linear_1", nn.Linear(64, 64))
        self.features.add_module("NetWorkSimple_LeakyReLU_1", nn.LeakyReLU())
        self.features.add_module("NetWorkSimple_Linear_Output", nn.Linear(64, 64))

        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, np.prod(self.action_space))
        )

        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def __str__(self):
        return 'SimpleDuelingNetwork-' + str(self.observation_space) + "-" + str(self.action_space)
