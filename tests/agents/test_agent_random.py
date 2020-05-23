import numpy as np
import pytest
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple

from torchforce.agents import AgentRandom


def test_agent_don_t_work_with_no_space():
    test_list = [1, 100, "100", "somethink", [], dict(), 0.0, 1245.215, None]
    for action_space in test_list:
        with pytest.raises(TypeError):
            AgentRandom(action_space=action_space)


base_list = {"box": Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32), "discrete": Discrete(3),
             "multibinary": MultiBinary(10), "multidiscrete": MultiDiscrete(10)}
dict_list = Dict(base_list)
tuple_list = Tuple(base_list.values())

test_list = [dict_list, tuple_list, *base_list.values()]


def test_agent_work_with_space():
    for action_space in test_list:
        AgentRandom(action_space=action_space)


def test_agent_get_action():
    for action_space in test_list:
        agent = AgentRandom(action_space=action_space)
        agent.get_action(None)


def test_agent_learn():
    for action_space in test_list:
        agent = AgentRandom(action_space=action_space)
        agent.learn(None, None, None, None)


def test_agent_episode_finished():
    for action_space in test_list:
        agent = AgentRandom(action_space=action_space)
        agent.episode_finished()
