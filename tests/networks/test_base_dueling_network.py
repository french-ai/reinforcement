import pytest
import torch
from gym.spaces import flatdim

from blobrl.networks import BaseDuelingNetwork
from tests.networks import TestBaseNetwork

TestBaseNetwork.__test__ = False


class TestBaseDuelingNetwork(TestBaseNetwork):
    __test__ = True

    network = BaseDuelingNetwork
