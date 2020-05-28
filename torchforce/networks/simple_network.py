import numpy as np
import torch.nn as nn

from torchforce.networks import BaseNetwork


class SimpleNetwork(BaseNetwork):
    def __init__(self, observation_shape, action_shape):
        super().__init__(observation_shape=observation_shape, action_shape=action_shape)

        if not isinstance(observation_shape, (tuple, int)):
            raise TypeError("observation_space need to be Space not " + str(type(observation_shape)))
        if not isinstance(action_shape, (tuple, int)):
            raise TypeError("action_space need to be Space not " + str(type(action_shape)))

        self.network = nn.Sequential()
        self.network.add_module("NetWorkSimple_Linear_Input", nn.Linear(np.prod(self.observation_space), 64))
        self.network.add_module("NetWorkSimple_LeakyReLU_Input", nn.LeakyReLU())
        self.network.add_module("NetWorkSimple_Linear_1", nn.Linear(64, 64))
        self.network.add_module("NetWorkSimple_LeakyReLU_1", nn.LeakyReLU())
        self.network.add_module("NetWorkSimple_Linear_Output", nn.Linear(64, np.prod(self.action_space)))

    def forward(self, observation):
        x = observation.view(observation.shape[0], -1)
        return self.network(x)
