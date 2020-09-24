import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Space, flatdim

from blobrl.networks import BaseNetwork


class C51Network(BaseNetwork):
    def __init__(self, observation_space, action_space):
        """

        :param observation_space:
        :param action_space:
        """
        super().__init__(observation_space=observation_space, action_space=action_space)

        self.NUM_ATOMS = 51

        self.network = nn.Sequential()
        self.network.add_module("C51_Linear_Input", nn.Linear(np.prod(flatdim(self.observation_space)), 64))
        self.network.add_module("C51_LeakyReLU_Input", nn.LeakyReLU())
        self.network.add_module("C51_Linear_1", nn.Linear(64, 64))
        self.network.add_module("C51_LeakyReLU_1", nn.LeakyReLU())

        self.distributional_list = []
        self.len_distributional = np.prod(flatdim(self.action_space))

        for i in range(self.len_distributional):
            distributional = nn.Sequential()
            distributional.add_module("C51_Distributional_" + str(i) + "_Linear", nn.Linear(64, self.NUM_ATOMS))
            distributional.add_module("C51_Distributional_" + str(i) + "_Softmax", nn.Softmax(dim=1))

            self.add_module("C51_Distributional_" + str(i) + "_Sequential", distributional)
            self.distributional_list.append(distributional)

    def forward(self, observation):
        """

        :param observation:
        :return:
        """
        x = observation.view(observation.shape[0], -1)
        x = self.network(x)

        q = [distributionalLayer(x) for distributionalLayer in self.distributional_list]
        q = torch.cat(q)
        q = torch.reshape(q, (self.len_distributional, -1, self.NUM_ATOMS))
        q = q.permute(1, 0, 2)

        return q

    def __str__(self):
        return 'C51Network-' + str(self.observation_space) + "-" + str(self.action_space)
