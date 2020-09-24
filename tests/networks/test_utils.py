from blobrl.networks import get_last_layers
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict
import torch.nn as nn


def valid_dim(out_v, out_g):
    if isinstance(out_v, list):
        assert len(out_v) == len(out_g)
        for o, g in zip(out_v, out_g):
            valid_dim(o, g)
    else:
        assert len(out_v.state_dict()) == len(out_g.state_dict())


def test_get_last_layers():
    in_values = [
        Discrete(10),
        Discrete(1),
        Discrete(100),
        Discrete(5),

        MultiDiscrete([1]),
        MultiDiscrete([10, 110, 3, 50]),
        MultiDiscrete([1, 1, 1]),
        MultiDiscrete([100, 3, 3, 5]),

        MultiDiscrete([[100, 3], [3, 5]]),
        MultiDiscrete([[[100, 3], [3, 5]], [[100, 3], [3, 5]]]),

        MultiBinary(1),
        MultiBinary(3),
        MultiBinary([3, 2]),

        Box(low=0, high=10, shape=[1]),
        Box(low=0, high=10, shape=[2, 2]),
        Box(low=0, high=10, shape=[2, 2, 2]),

        Tuple([Discrete(1), MultiDiscrete([1, 1])]),
        Dict({"first": Discrete(1), "second": MultiDiscrete([1, 1])})

    ]

    out_values = [

        nn.Linear(10, 10),
        nn.Linear(10, 1),
        nn.Linear(10, 100),
        nn.Linear(10, 5),

        [nn.Sequential(*[nn.Linear(10, 10), nn.Softmax()])],
        [nn.Sequential(*[nn.Linear(10, 10), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 110), nn.Softmax()]),
         nn.Sequential(*[nn.Linear(10, 3), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 50), nn.Softmax()])],
        [nn.Sequential(*[nn.Linear(10, 1), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 1), nn.Softmax()]),
         nn.Sequential(*[nn.Linear(10, 1), nn.Softmax()])],
        [nn.Sequential(*[nn.Linear(10, 100), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 3), nn.Softmax()]),
         nn.Sequential(*[nn.Linear(10, 3), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 5), nn.Softmax()])],

        [[nn.Sequential(*[nn.Linear(10, 100), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 3), nn.Softmax()])],
         [nn.Sequential(*[nn.Linear(10, 3), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 5), nn.Softmax()])]],
        [
            [[nn.Sequential(*[nn.Linear(10, 100), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 3), nn.Softmax()])],
             [nn.Sequential(*[nn.Linear(10, 3), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 5), nn.Softmax()])]],
            [[nn.Sequential(*[nn.Linear(10, 100), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 3), nn.Softmax()])],
             [nn.Sequential(*[nn.Linear(10, 3), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 5), nn.Softmax()])]]
        ],

        [nn.Sequential(*[nn.Linear(10, 1), nn.Sigmoid()])],
        [nn.Sequential(*[nn.Linear(10, 1), nn.Sigmoid()]),
         nn.Sequential(*[nn.Linear(10, 1), nn.Sigmoid()]),
         nn.Sequential(*[nn.Linear(10, 1), nn.Sigmoid()])],

        [
            [nn.Sequential(*[nn.Linear(10, 1), nn.Sigmoid()]),
             nn.Sequential(*[nn.Linear(10, 1), nn.Sigmoid()])],
            [nn.Sequential(*[nn.Linear(10, 1), nn.Sigmoid()]),
             nn.Sequential(*[nn.Linear(10, 1), nn.Sigmoid()])],
            [nn.Sequential(*[nn.Linear(10, 1), nn.Sigmoid()]),
             nn.Sequential(*[nn.Linear(10, 1), nn.Sigmoid()])]
        ],

        [nn.Linear(10, 1)]
        ,
        [[nn.Linear(10, 1), nn.Linear(10, 1)], [nn.Linear(10, 1), nn.Linear(10, 1)]]
        ,
        [[[nn.Linear(10, 1), nn.Linear(10, 1)], [nn.Linear(10, 1), nn.Linear(10, 1)]],
         [[nn.Linear(10, 1), nn.Linear(10, 1)], [nn.Linear(10, 1), nn.Linear(10, 1)]]],

        [nn.Linear(10, 1),
         [nn.Sequential(*[nn.Linear(10, 1), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 1), nn.Softmax()])]],

        [nn.Linear(10, 1),
         [nn.Sequential(*[nn.Linear(10, 1), nn.Softmax()]), nn.Sequential(*[nn.Linear(10, 1), nn.Softmax()])]],

    ]

    for in_value, out_value in zip(in_values, out_values):
        print(in_value)
        out_value_gen = get_last_layers(in_value, 10)
        valid_dim(out_value, out_value_gen)
