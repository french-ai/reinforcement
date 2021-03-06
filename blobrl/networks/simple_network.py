import numpy as np
import torch.nn as nn
from gym.spaces import flatdim
from .utils import get_last_layers
from blobrl.networks import BaseNetwork


class SimpleNetwork(BaseNetwork):
    def __init__(self, observation_space, action_space, linear_dim=64):
        """

        :param observation_space:
        :param action_space:
        """
        super().__init__(observation_space=observation_space, action_space=action_space)

        self.network = nn.Sequential()
        self.network.add_module("NetWorkSimple_Linear_Input",
                                nn.Linear(np.prod(flatdim(self.observation_space)), linear_dim))
        self.network.add_module("NetWorkSimple_LeakyReLU_Input", nn.LeakyReLU())
        self.network.add_module("NetWorkSimple_Linear_1", nn.Linear(linear_dim, linear_dim))
        self.network.add_module("NetWorkSimple_LeakyReLU_1", nn.LeakyReLU())

        self.outputs = get_last_layers(self.action_space, last_dim=linear_dim)

    def forward(self, observation):
        """

        :param observation:
        :return:
        """

        x = observation.view(observation.shape[0], -1)
        x = self.network(x)

        def forwards(last_tensor, layers):
            if isinstance(layers, list):
                return [forwards(last_tensor, layers) for layers in layers]
            return layers(last_tensor)

        return forwards(x, self.outputs)

    def __str__(self):
        return 'SimpleNetwork-' + str(self.observation_space) + "-" + str(self.action_space)
