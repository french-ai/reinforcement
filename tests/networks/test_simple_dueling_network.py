import pytest
import torch
from gym.spaces import Discrete, flatdim

from blobrl.networks import SimpleDuelingNetwork


def test_simple_network_init():
    list_fail = [[None, None],
                 ["dedrfe", "qdzq"],
                 [1215.4154, 157.48],
                 ["zdzd", (Discrete(1))],
                 [Discrete(1), "zdzd"],
                 ["zdzd", (1, 4, 7)],
                 [(1, 4, 7), "zdzd"],
                 [152, 485]]

    for ob, ac in list_fail:
        with pytest.raises(TypeError):
            SimpleDuelingNetwork(observation_space=ob, action_space=ac)

    list_work = [[Discrete(1), Discrete(1)],
                 [Discrete(10), Discrete(10)],
                 [Discrete(1), Discrete(50)]]
    for ob, ac in list_work:
        SimpleDuelingNetwork(observation_space=ob, action_space=ac)


def test_forward():
    list_work = [[Discrete(1), Discrete(1)],
                 [Discrete(10), Discrete(10)],
                 [Discrete(1), Discrete(50)]]
    for ob, ac in list_work:
        simple_network = SimpleDuelingNetwork(observation_space=ob, action_space=ac)
        simple_network.forward(torch.rand((1, flatdim(ob))))


def test__str__():
    list_work = [[Discrete(1), Discrete(1)],
                 [Discrete(10), Discrete(10)],
                 [Discrete(1), Discrete(50)]]

    for ob, ac in list_work:
        network = SimpleDuelingNetwork(observation_space=ob, action_space=ac)

        assert 'SimpleDuelingNetwork-' + str(ob) + "-" + str(ac) == network.__str__()
