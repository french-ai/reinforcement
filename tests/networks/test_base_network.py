import pytest

from blobrl.networks import BaseNetwork
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box, Tuple, Dict


class TestBaseNetwork:
    __test__ = True

    network = BaseNetwork

    list_work = [
        [Discrete(3), Discrete(1)],
        [Discrete(3), Discrete(3)],
        [Discrete(10), Discrete(50)],
        [MultiDiscrete([3]), MultiDiscrete([1])],
        [MultiDiscrete([3, 3]), MultiDiscrete([3, 3])],
        [MultiDiscrete([4, 4, 4]), MultiDiscrete([50, 4, 4])],
        [MultiDiscrete([[100, 3], [3, 5]]), MultiDiscrete([[100, 3], [3, 5]])],
        [MultiDiscrete([[[100, 3], [3, 5]], [[100, 3], [3, 5]]]),
         MultiDiscrete([[[100, 3], [3, 5]], [[100, 3], [3, 5]]])],
        [MultiBinary(1), MultiBinary(1)],
        [MultiBinary(3), MultiBinary(3)],
        # [MultiBinary([3, 2]), MultiBinary([3, 2])], # Don't work yet because gym don't implemented this
        [Box(low=0, high=10, shape=[1]), Box(low=0, high=10, shape=[1])],
        [Box(low=0, high=10, shape=[2, 2]), Box(low=0, high=10, shape=[2, 2])],
        [Box(low=0, high=10, shape=[2, 2, 2]), Box(low=0, high=10, shape=[2, 2, 2])],

        [Tuple([Discrete(1), MultiDiscrete([1, 1])]), Tuple([Discrete(1), MultiDiscrete([1, 1])])],
        [Dict({"first": Discrete(1), "second": MultiDiscrete([1, 1])}),
         Dict({"first": Discrete(1), "second": MultiDiscrete([1, 1])})],

    ]
    list_fail = [
        [None, None],
        ["dedrfe", "qdzq"],
        [1215.4154, 157.48],
        ["zdzd", (Discrete(1))],
        [Discrete(1), "zdzd"],
        ["zdzd", (1, 4, 7)],
        [(1, 4, 7), "zdzd"],
        [152, 485]
    ]

    def test_init(self):
        for ob, ac in self.list_fail:
            with pytest.raises(TypeError):
                self.network(observation_space=ob, action_space=ac)

        for ob, ac in self.list_work:
            with pytest.raises(TypeError):
                self.network(observation_space=ob, action_space=ac)
