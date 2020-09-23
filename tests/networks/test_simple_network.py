import pytest
import torch
from gym.spaces import Discrete, flatdim
from blobrl.networks import SimpleNetwork


def test_simple_network_init():
    list_fail = [[None, None],
                 ["dedrfe", "qdzq"],
                 [1215.4154, 157.48],
                 ["zdzd", (Discrete(1))],
                 [Discrete(1), "zdzd"],
                 ["zdzd", (1, 4, 7)],
                 [(1, 4, 7), "zdzd"],
                 ]

    for ob, ac in list_fail:
        with pytest.raises(TypeError):
            SimpleNetwork(observation_space=ob, action_space=ac)

    list_work = [[Discrete(1), Discrete(1)],
                 [Discrete(2), Discrete(2)],
                 [Discrete(1), Discrete(1)]]
    for ob, ac in list_work:
        SimpleNetwork(observation_space=ob, action_space=ac)


def test_forward():
    list_work = [[Discrete(3), Discrete(1)],
                 [Discrete(3), Discrete(3)]]
    for ob, ac in list_work:
        simple_network = SimpleNetwork(observation_space=ob, action_space=ac)
        simple_network.forward(torch.rand((1, flatdim(ob))))
