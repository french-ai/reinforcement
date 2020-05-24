import gym
import pytest

from torchforce import Trainer
from torchforce.agents import AgentInterface


class FakeEnv(gym.Env):

    def __init__(self):
        super(FakeEnv).__init__()
        self.step_done = False
        self.reset = False
        self.render = False

    def step(self, action):
        self.step_done = True
        return None, 0, False, None

    def reset(self):
        self.reset = True
        return None

    def render(self, mode='human'):
        self.render = True


class FakeAgent(AgentInterface):

    def __init__(self):
        super(FakeEnv).__init__()
        self.get_action_done = False
        self.learn = False
        self.episode_finished = False

    def get_action(self, observation):
        self.get_action_done = True
        return None

    def learn(self, observation, action, reward, next_observation) -> None:
        self.learn = True

    def episode_finished(self) -> None:
        self.episode_finished = True


def test_arg_to_agent():
    fail_list = ["dzdzqd", None, 123, 123.123, [], {}]
    work_list = ["agent_random", FakeAgent()]

    for agent in fail_list:
        with pytest.raises(ValueError):
            Trainer.arg_to_agent(agent)

    for agent in work_list:
        Trainer.arg_to_agent(agent)


def test_get_agent():
    fail_list = ["dzdzqd", None, 123, 123.123, [], {}]
    work_list = ["CartPole-v1", FakeEnv()]

    for env in fail_list:
        with pytest.raises(ValueError):
            Trainer.get_environment(env)

    for env in work_list:
        Trainer.get_environment(env)


def test_do_episode():
    fake_env = FakeEnv()
    fake_agent = FakeAgent()
    assert fake_agent.get_action_done is False and fake_agent.learn is False and fake_agent.episode_finished is False
    assert fake_env.step_done is False and fake_env.reset is False and fake_env.render is False

    Trainer.do_episode(env=fake_env, agent=fake_agent)
    assert fake_agent.get_action_done is True and fake_agent.learn is True and fake_agent.episode_finished is True
    assert fake_env.step_done is True and fake_env.reset is True and fake_env.render is True
