import pytest
import torch
from gym.spaces import Discrete, flatdim

from blobrl.networks import C51Network


def test_c51_init():
    list_fail = [[None, None],
                 ["dedrfe", "qdzq"],
                 [1215.4154, 157.48],
                 ["zdzd", (Discrete(1))],
                 [Discrete(1), "zdzd"],
                 ["zdzd", (1, 4, 7)],
                 [(1, 4, 7), "zdzd"],
                 [15, 24]]

    for ob, ac in list_fail:
        with pytest.raises(TypeError):
            C51Network(observation_space=ob, action_space=ac)

    list_work = [[Discrete(1), Discrete(1)],
                 [Discrete(10), Discrete(20)],
                 [Discrete(1), Discrete(20)]]
    for ob, ac in list_work:
        C51Network(observation_space=ob, action_space=ac)


def test_c51_forward():
    list_work = [[Discrete(1), Discrete(1)],
                 [Discrete(10), Discrete(20)],
                 [Discrete(1), Discrete(20)]]
    for ob, ac in list_work:
        simple_network = C51Network(observation_space=ob, action_space=ac)
        simple_network.forward(torch.rand((2, flatdim(ob))))
