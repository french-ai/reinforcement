from gym.spaces import flatdim
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict
import torch.nn as nn
import numpy as np


def get_last_layers(space, last_dim):
    if isinstance(space, Box):
        def map_box(ld):
            def mp(n):
                if isinstance(n, list):
                    return [mp(x) for x in n]
                return nn.Linear(ld, 1)

            return mp

        return list(map(map_box(last_dim), np.empty(space.shape).tolist()))
    elif isinstance(space, Discrete):
        return nn.Sequential(*[nn.Linear(last_dim, flatdim(space)), nn.Softmax()])
    elif isinstance(space, Tuple):
        return [get_last_layers(s, last_dim) for s in space]
    elif isinstance(space, Dict):
        return [get_last_layers(s, last_dim) for s in space.spaces.values()]
    elif isinstance(space, MultiBinary):
        def map_multibinary(ld):
            def mp(n):
                if isinstance(n, list):
                    return [mp(x) for x in n]

                return nn.Sequential(*[nn.Linear(ld, 1), nn.Sigmoid()])

            return mp

        return list(map(map_multibinary(last_dim), np.empty(space.n).tolist()))
    elif isinstance(space, MultiDiscrete):
        def map_multidiscrete(ld):
            def mp(n):
                if isinstance(n, list):
                    return [mp(x) for x in n]

                return nn.Sequential(*[nn.Linear(ld, n), nn.Softmax()])

            return mp

        return list(map(map_multidiscrete(last_dim), space.nvec.tolist()))
    else:
        raise NotImplementedError
