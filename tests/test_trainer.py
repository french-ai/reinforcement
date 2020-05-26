import gym
import pytest

from torchforce import Trainer, Logger
from torchforce.agents import AgentInterface


class FakeEnv(gym.Env):

    def __init__(self):
        super(FakeEnv).__init__()
        self.step_done = 0
        self.reset_done = 0
        self.render_done = 0

    def step(self, action):
        self.step_done += 1
        return True, 0, True, True

    def reset(self):
        self.reset_done += 1
        return True

    def render(self, mode='human'):
        self.render_done += 1


class FakeAgent(AgentInterface):

    def __init__(self):
        super(FakeAgent).__init__()
        self.get_action_done = 0
        self.learn_done = 0
        self.episode_finished_done = 0

    def get_action(self, observation):
        self.get_action_done += 1
        return True

    def learn(self, observation, action, reward, next_observation) -> None:
        self.learn_done += 1

    def episode_finished(self) -> None:
        self.episode_finished_done += 1


class FakeLogger(Logger):
    def __init__(self):
        self.add_steps_call = 0
        self.add_episode_call = 0
        self.end_episode_call = 0

    def add_steps(self, steps):
        self.add_steps_call += 1

    def add_episode(self, episode):
        self.add_episode_call += 1

    def end_episode(self):
        self.end_episode_call += 1


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
    logger = FakeLogger()

    assert fake_agent.episode_finished_done == 0
    assert fake_env.reset_done == 0

    Trainer.do_episode(env=fake_env, agent=fake_agent)
    assert fake_agent.episode_finished_done == 1
    assert fake_env.reset_done == 1
    assert logger.add_steps_call == 0 and logger.add_episode_call == 0 and logger.end_episode_call == 0

    Trainer.do_episode(env=fake_env, agent=fake_agent, logger=logger)
    assert fake_agent.episode_finished_done == 2
    assert fake_env.reset_done == 2
    assert logger.add_steps_call == 1 and logger.add_episode_call == 0 and logger.end_episode_call == 1


def test_do_step():
    fake_env = FakeEnv()
    fake_agent = FakeAgent()
    logger = FakeLogger()

    assert fake_agent.get_action_done == 0 and fake_agent.learn_done == 0 and fake_agent.episode_finished_done == 0
    assert fake_env.step_done == 0 and fake_env.reset_done == 0 and fake_env.render_done == 0

    Trainer.do_step(observation=None, env=fake_env, agent=fake_agent)
    assert fake_agent.get_action_done == 1 and fake_agent.learn_done == 1 and fake_agent.episode_finished_done == 0
    assert fake_env.step_done == 1 and fake_env.reset_done == 0 and fake_env.render_done == 1
    assert logger.add_steps_call == 0 and logger.add_episode_call == 0 and logger.end_episode_call == 0

    Trainer.do_step(observation=None, env=fake_env, agent=fake_agent, logger=logger)
    assert fake_agent.get_action_done == 2 and fake_agent.learn_done == 2 and fake_agent.episode_finished_done == 0
    assert fake_env.step_done == 2 and fake_env.reset_done == 0 and fake_env.render_done == 2
    assert logger.add_steps_call == 1 and logger.add_episode_call == 0 and logger.end_episode_call == 0


def test_init_trainer():
    trainer = Trainer(environment=FakeEnv(), agent=FakeAgent())
    assert isinstance(trainer.agent, AgentInterface)
    assert isinstance(trainer.environment, gym.Env)
    assert isinstance(trainer.logger, Logger)

    trainer = Trainer(environment="CartPole-v1", agent=FakeAgent())
    assert isinstance(trainer.agent, AgentInterface)
    assert isinstance(trainer.environment, str)
    assert isinstance(trainer.logger, Logger)

    with pytest.raises(ValueError):
        Trainer(environment="CartPole-v1", agent="dzdqsdz")


def test_trainer_train():
    test_list = [0, 1, 10, 100, 1000]

    for number_episode in test_list:
        fake_env = FakeEnv()
        fake_agent = FakeAgent()
        trainer = Trainer(environment=fake_env, agent=fake_agent)

        assert fake_agent.get_action_done == 0 and fake_agent.learn_done == 0 and fake_agent.episode_finished_done == 0
        assert fake_env.step_done == 0 and fake_env.reset_done == 0 and fake_env.render_done == 0

        trainer.train(max_episode=number_episode)

        assert fake_agent.get_action_done == number_episode and fake_agent.learn_done == number_episode
        assert fake_agent.episode_finished_done == number_episode
        assert fake_env.step_done == number_episode and fake_env.reset_done == number_episode
        assert fake_env.render_done == number_episode
