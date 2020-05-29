import os

import numpy as np
import pytest
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple

from torchforce.agents import AgentRandom


def test_agent_don_t_work_with_no_space():
    test_list = [1, 100, "100", "somethink", [], dict(), 0.0, 1245.215, None]
    for action_space in test_list:
        with pytest.raises(TypeError):
            AgentRandom(observation_space=Discrete(1), action_space=action_space)

    for observation_space in test_list:
        with pytest.raises(TypeError):
            AgentRandom(observation_space=observation_space, action_space=Discrete(1))


base_list = {"box": Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32), "discrete": Discrete(3),
             "multibinary": MultiBinary(10), "multidiscrete": MultiDiscrete(10)}
dict_list = Dict(base_list)
tuple_list = Tuple(list(base_list.values()))

test_list = [*base_list.values(), dict_list, tuple_list]


def test_agent_work_with_space():
    for space in test_list:
        agent = AgentRandom(observation_space=space, action_space=space)
        assert agent.observation_space == space
        assert agent.action_space == space


def test_agent_get_action():
    for space in test_list:
        agent = AgentRandom(observation_space=space, action_space=space)
        agent.get_action(None)


def test_agent_learn():
    for space in test_list:
        agent = AgentRandom(observation_space=space, action_space=space)
        agent.learn(None, None, None, None, None)


def test_agent_episode_finished():
    for space in test_list:
        agent = AgentRandom(observation_space=space, action_space=space)
        agent.episode_finished()


def test_agent_save_load():
    for space in test_list:
        agent = AgentRandom(observation_space=space, action_space=space)

        agent.save(file_name="deed.pt")
        agent_l = AgentRandom.load(file_name="deed.pt")

        assert agent.observation_space == agent_l.observation_space
        assert agent.action_space == agent_l.action_space
        os.remove("deed.pt")

    agent = AgentRandom(observation_space=space, action_space=space)
    agent.save(file_name="deed.pt", dire_name="./remove/")

    os.remove("./remove/deed.pt")
    os.rmdir("./remove/")

    with pytest.raises(TypeError):
        agent.save(file_name=14548)
    with pytest.raises(TypeError):
        agent.save(file_name="deed.pt", dire_name=14484)

    with pytest.raises(FileNotFoundError):
        AgentRandom.load(file_name="deed.pt")
    with pytest.raises(FileNotFoundError):
        AgentRandom.load(file_name="deed.pt", dire_name="/Dede/")
