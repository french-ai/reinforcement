import numpy as np
import torch
import torch.nn as nn
from gym.spaces import flatdim, Discrete, MultiDiscrete

from blobrl.networks import BaseNetwork


class C51Network(BaseNetwork):
    def __init__(self, observation_space, action_space):
        """

        :param observation_space:
        :param action_space:
        """
        if not isinstance(action_space, (Discrete, MultiDiscrete)):
            raise TypeError(
                "action_space need to be instance of Discrete or MultiDiscrete, not :" + str(type(action_space)))

        super().__init__(observation_space=observation_space, action_space=action_space)

        self.NUM_ATOMS = 51

        self.network = nn.Sequential()
        self.network.add_module("C51_Linear_Input", nn.Linear(np.prod(flatdim(self.observation_space)), 64))
        self.network.add_module("C51_LeakyReLU_Input", nn.LeakyReLU())
        self.network.add_module("C51_Linear_1", nn.Linear(64, 64))
        self.network.add_module("C51_LeakyReLU_1", nn.LeakyReLU())

        self.distributional_list = []
        if isinstance(self.action_space, Discrete):
            self.len_distributional = self.action_space.n

            for i in range(self.len_distributional):
                distributional = nn.Sequential()
                distributional.add_module("C51_Distributional_" + str(i) + "_Linear", nn.Linear(64, self.NUM_ATOMS))
                distributional.add_module("C51_Distributional_" + str(i) + "_Softmax", nn.Softmax(dim=1))

                self.add_module("C51_Distributional_" + str(i) + "_Sequential", distributional)
                self.distributional_list.append(distributional)

        elif isinstance(self.action_space, MultiDiscrete):
            def gen_outputs(nvec):
                dis = []
                for nspace in nvec:
                    if isinstance(nspace, (list, np.ndarray)):
                        dis.append(gen_outputs(nspace))
                    else:
                        dis.append(
                            [nn.Sequential(nn.Linear(64, self.NUM_ATOMS), nn.Softmax(dim=1)) for i in range(nspace)])
                return dis

            self.distributional_list = gen_outputs(self.action_space.nvec)

    def forward(self, observation):
        """

        :param observation:
        :return:
        """
        x = observation.view(observation.shape[0], -1)
        x = self.network(x)
        if isinstance(self.action_space, Discrete):
            q = [distributionalLayer(x) for distributionalLayer in self.distributional_list]
            q = torch.cat(q)
            q = torch.reshape(q, (self.action_space.n, -1, self.NUM_ATOMS))
            q = q.permute(1, 0, 2)

            return q
        elif isinstance(self.action_space, MultiDiscrete):

            def do_forward(nvec, llayers, x):
                if isinstance(llayers[-1], list):
                    return [do_forward(n, l, x) for n, l in zip(nvec, llayers)]

                q = [distributionalLayer(x) for distributionalLayer in llayers]
                q = torch.cat(q)
                q = torch.reshape(q, (nvec, -1, self.NUM_ATOMS))
                q = q.permute(1, 0, 2)

                return q

            return do_forward(self.action_space.nvec, self.distributional_list, x)

    def __str__(self):
        return 'C51Network-' + str(self.observation_space) + "-" + str(self.action_space)
