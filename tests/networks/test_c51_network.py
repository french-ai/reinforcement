import pytest
import torch
from gym.spaces import Discrete, Box

from torchforce.networks import C51Network


def test_c51_init():
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
            C51Network(observation_shape=ob, action_shape=ac)

    list_work = [[(454), (4)],
                 [(454, 54), (5, 2)],
                 [(454, 54, 45), (4, 5, 3)]]
    for ob, ac in list_work:
        C51Network(observation_shape=ob, action_shape=ac)


def test_c51_forward():
    list_work = [[(454, 54), (4, 2)],
                 [(454, 54, 45), (5, 1, 2)]]
    for ob, ac in list_work:
        simple_network = C51Network(observation_shape=ob, action_shape=ac)
        simple_network.forward(torch.rand((2, *ob)))
