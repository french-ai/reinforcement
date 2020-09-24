import pytest
import torch
from gym.spaces import flatdim

from blobrl.networks import SimpleDuelingNetwork
from tests.networks import TestBaseDuelingNetwork

TestBaseDuelingNetwork.__test__ = False


class TestSimpleDuelingNetwork(TestBaseDuelingNetwork):
    __test__ = True

    network = SimpleDuelingNetwork

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

            assert 'SimpleDuelingNetwork-' + str(ob) + "-" + str(ac) == network.__str__()
