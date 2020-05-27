import numpy as np
import torch
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
        self.network.add_module("NetWorkSimple_Linear_Input", nn.Linear(np.prod(self.observation_space), 256))
        self.network.add_module("NetWorkSimple_Linear_1", nn.Linear(256, 256))
        self.network.add_module("NetWorkSimple_Linear_2", nn.Linear(256, 256))
        self.network.add_module("NetWorkSimple_Linear_Ouput", nn.Linear(256, np.prod(self.observation_space)))

    def forward(self, observation):
        x = torch.flatten(observation)
        return self.network(x)
