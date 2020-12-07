import pytest
import torch
from gym.spaces import flatdim, flatten

from blobrl.networks import SimpleDuelingNetwork, SimpleNetwork
from tests.networks import TestBaseDuelingNetwork


class TestSimpleDuelingNetwork(TestBaseDuelingNetwork):
    __test__ = True

    network = SimpleDuelingNetwork
    net = SimpleNetwork

    list_fail = [1,
                 0.1,
                 "string",
                 object(),
                 network(net(TestBaseDuelingNetwork.list_work[0][0], TestBaseDuelingNetwork.list_work[0][1]))
                 ]

    def test_init(self):
        for net in self.list_fail:
            with pytest.raises(TypeError):
                self.network(net)

        for ob, ac in self.list_work:
            self.network(self.net(observation_space=ob, action_space=ac))

    def test_forward(self):
        for ob, ac in self.list_work:
            network = self.network(self.net(observation_space=ob, action_space=ac))
            network.forward(torch.rand((1, flatdim(ob))))

    def test_str_(self):
        for ob, ac in self.list_work:
            net = self.net(observation_space=ob, action_space=ac)
            network = self.network(net)

            assert 'SimpleDuelingNetwork-' + str(net) == network.__str__()

    def test_call_network(self):
        for ob, ac in self.list_work:
            self.network(SimpleNetwork(observation_space=ob, action_space=ac))(
                torch.tensor([flatten(ob, ob.sample())]).float())
