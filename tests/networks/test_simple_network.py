import pytest
import torch
from gym.spaces import flatdim, flatten

from blobrl.networks import SimpleNetwork
from tests.networks import TestBaseNetwork

TestBaseNetwork.__test__ = False


class TestSimpleNetwork(TestBaseNetwork):
    __test__ = True

    network = SimpleNetwork

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

            assert 'SimpleNetwork-' + str(ob) + "-" + str(ac) == network.__str__()

    def test_call_network(self):
        for ob, ac in self.list_work:
            self.network(observation_space=ob, action_space=ac)(
                torch.tensor([flatten(ob, ob.sample())]).float())
