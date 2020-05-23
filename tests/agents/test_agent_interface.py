import pytest

from torchforce.agents import AgentInterface


def test_can_t_instantiate_agent_interface():
    with pytest.raises(TypeError):
        AgentInterface()
