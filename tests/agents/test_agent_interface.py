import pytest
import torch
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Dict, Tuple, Box

from blobrl.agents import AgentInterface


class MOCKAgentInterface(AgentInterface):
    def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)

    def get_action(self, observation):
        pass

    def enable_exploration(self):
        pass

    def disable_exploration(self):
        pass

    def learn(self, observation, action, reward, next_observation, done) -> None:
        pass

    def episode_finished(self) -> None:
        pass

    def save(self, file_name, dire_name="."):
        pass

    @classmethod
    def load(cls, file_name, dire_name=".", device=None):
        pass

    def __str__(self):
        return ""


class TestAgentInterface:
    __test__ = True

    agent = MOCKAgentInterface

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
        for o, a in self.list_work:
            with pytest.raises(TypeError):
                self.agent(o, a, "cpu")

        for o, a in self.list_fail:
            with pytest.raises(TypeError):
                self.agent(o, a, "cpu")

    def test_device(self):
        for o, a in self.list_work:
            device = torch.device("cpu")
            assert device == self.agent(o, a, device).device

            device = None
            assert torch.device("cpu") == self.agent(o, a, device).device

            for device in ["dzeqdzqd", 1512, object(), 151.515]:
                with pytest.raises(TypeError):
                    self.agent(o, a, device)

            if torch.cuda.is_available():
                self.agent(o, a, torch.device("cuda"))

    def test__str__(self):

        pass
