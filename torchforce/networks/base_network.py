import abc

import torch.nn as nn


class BaseNetwork(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.observation_space = observation_shape
        self.action_space = action_shape

    @abc.abstractmethod
    def __str__(self):
        pass
