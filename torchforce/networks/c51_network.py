import numpy as np
import torch
import torch.nn as nn

from torchforce.networks import BaseNetwork


class C51Network(BaseNetwork):
    def __init__(self, observation_shape, action_shape):
        super().__init__(observation_shape=observation_shape, action_shape=action_shape)

        if not isinstance(observation_shape, (tuple, int)):
            raise TypeError("observation_space need to be Space not " + str(type(observation_shape)))
        if not isinstance(action_shape, (tuple, int)):
            raise TypeError("action_space need to be Space not " + str(type(action_shape)))

        self.NUM_ATOMS = 51

        self.network = nn.Sequential()
        self.network.add_module("C51_Linear_Input", nn.Linear(np.prod(self.observation_space), 64))
        self.network.add_module("C51_LeakyReLU_Input", nn.LeakyReLU())
        self.network.add_module("C51_Linear_1", nn.Linear(64, 64))
        self.network.add_module("C51_LeakyReLU_1", nn.LeakyReLU())

        self.distributional_list = []
        self.len_distributional = np.prod(self.action_space)

        for i in range(self.len_distributional):
            distributional = nn.Sequential()
            distributional.add_module("C51_Distributional_" + str(i) + "_Linear", nn.Linear(64, self.NUM_ATOMS))
            distributional.add_module("C51_Distributional_" + str(i) + "_Softmax", nn.Softmax(dim=1))

            self.distributional_list.append(distributional)

    def forward(self, observation):
        x = observation.view(observation.shape[0], -1)
        x = self.network(x)

        q = [distributionalLayer(x) for distributionalLayer in self.distributional_list]
        q = torch.cat(q)
        q = torch.reshape(q, (self.len_distributional, -1, self.NUM_ATOMS))
        q = q.permute(1, 0, 2)

        return q
