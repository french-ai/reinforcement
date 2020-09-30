import pytest
import torch
import numpy as np
from gym.spaces import flatdim

from blobrl.networks import C51Network
from tests.networks import TestBaseNetwork

TestBaseNetwork.__test__ = False


class TestC51Network(TestBaseNetwork):
    __test__ = True

    network = C51Network

    def test_init(self):
        for ob, ac in self.list_fail:
            with pytest.raises(TypeError):
                self.network(observation_space=ob, action_space=ac)

        for ob, ac in self.list_work:
            self.network(observation_space=ob, action_space=ac)

    def test_forward(self):
        for ob, ac in self.list_work:
            network = self.network(observation_space=ob, action_space=ac)
            network.forward(torch.rand((1, flatdim(ob))))

    def test_str_(self):
        for ob, ac in self.list_work:
            network = self.network(observation_space=ob, action_space=ac)

            assert 'C51Network-' + str(ob) + "-" + str(ac) == network.__str__()

    def test_call_network(self):
        for ob, ac in self.list_work:
            self.network(observation_space=ob, action_space=ac)(torch.from_numpy(np.array(ob.sample())))
