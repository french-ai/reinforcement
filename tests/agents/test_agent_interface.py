import pytest
import torch

from torchforce.agents import AgentInterface


def test_can_t_instantiate_agent_interface():
    with pytest.raises(TypeError):
        AgentInterface()


class MOCKAgentInterface(AgentInterface):
    def __init__(self, device):
        super().__init__(device)

    def get_action(self, observation):
        pass

    def enable_train(self):
        pass

    def disable_train(self):
        pass

    def learn(self, observation, action, reward, next_observation, done) -> None:
        pass

    def episode_finished(self) -> None:
        pass

    def save(self, file_name, dire_name="."):
        pass

    @classmethod
    def load(cls, file_name, dire_name="."):
        pass

    def __str__(self):
        return ""


def test_device():
    device = torch.device
    assert device == MOCKAgentInterface(device).device
