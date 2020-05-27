import pytest
import torch
from gym.spaces import Discrete

from torchforce.networks import SimpleNetwork


def test_simple_network_init():
    list_fail = [[None, None],
                 ["dedrfe", "qdzq"],
                 [1215.4154, 157.48],
                 ["zdzd", (Discrete(1))],
                 [Discrete(1), "zdzd"],
                 ["zdzd", (1, 4, 7)],
                 [(1, 4, 7), "zdzd"],
                 [Discrete(1), Discrete(1)]]

    for ob, ac in list_fail:
        with pytest.raises(TypeError):
            SimpleNetwork(observation_shape=ob, action_shape=ac)

    list_work = [[(454), (874)],
                 [(454, 54), (48, 44)],
                 [(454, 54, 45), (48, 44, 47)]]
    for ob, ac in list_work:
        SimpleNetwork(observation_shape=ob, action_shape=ac)


def test_forward():
    list_work = [[(454), (874)],
                 [(454, 54), (48, 44)],
                 [(454, 54, 45), (48, 44, 47)]]
    for ob, ac in list_work:
        simple_network = SimpleNetwork(observation_shape=ob, action_shape=ac)
        simple_network.forward(torch.rand(ob))
