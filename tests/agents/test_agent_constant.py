import os
import pytest

from blobrl.agents import AgentConstant
from tests.agents import TestAgentInterface

TestAgentInterface.__test__ = False


class TestAgentConstant(TestAgentInterface):
    __test__ = True

    agent = AgentConstant

    def test_init(self):
        for o, a in self.list_work:
            self.agent(o, a)

        for o, a in self.list_fail:
            with pytest.raises(TypeError):
                self.agent(o, a)

    def test_get_action(self):
        for o, a in self.list_work:
            agent = self.agent(observation_space=o, action_space=a)
            agent.get_action(None)

    def test_learn(self):
        for o, a in self.list_work:
            agent = self.agent(observation_space=o, action_space=a)
            agent.learn(None, None, None, None, None)

    def test_episode_finished(self):
        for o, a in self.list_work:
            agent = self.agent(observation_space=o, action_space=a)
            agent.episode_finished()

    def test_agent_save_load(self):
        for o, a in self.list_work:
            agent = self.agent(observation_space=o, action_space=a)

            agent.save(file_name="deed.pt")
            agent_l = self.agent.load(file_name="deed.pt")

            assert agent.observation_space == agent_l.observation_space
            assert agent.action_space == agent_l.action_space
            os.remove("deed.pt")

            agent = self.agent(observation_space=o, action_space=a)
            agent.save(file_name="deed.pt", dire_name="./remove/")

            os.remove("./remove/deed.pt")
            os.rmdir("./remove/")

            with pytest.raises(TypeError):
                agent.save(file_name=14548)
            with pytest.raises(TypeError):
                agent.save(file_name="deed.pt", dire_name=14484)

            with pytest.raises(FileNotFoundError):
                self.agent.load(file_name="deed.pt")
            with pytest.raises(FileNotFoundError):
                self.agent.load(file_name="deed.pt", dire_name="/Dede/")

    def test__str__(self):
        for o, a in self.list_work:
            agent = self.agent(observation_space=o, action_space=a)

            assert 'AgentConstant-' + str(o) + "-" + str(a) + "-" + str(agent.action) == agent.__str__()
