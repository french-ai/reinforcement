import platform
from shutil import rmtree

import gym
import matplotlib.pyplot as plt
import numpy as np
import pytest

from torchforce import Trainer, Logger
from torchforce.agents import AgentInterface
from torchforce.trainer import arg_to_agent


def test_arg_to_agent():
    fail_list = ["dzdzqd", None, 123, 123.123, [], {}, object]
    work_list = ["agent_random", "dqn", "double_dqn", "categorical_dqn", "dueling_dqn"]

    for agent in fail_list:
        with pytest.raises(ValueError):
            arg_to_agent(agent)

    for agent in work_list:
        arg_to_agent(agent)


class FakeEnv(gym.Env):

    def __init__(self):
        super(FakeEnv).__init__()
        self.step_done = 0
        self.reset_done = 0
        self.render_done = 0
        self.close_done = 0

    def step(self, action):
        self.step_done += 1
        return True, 0, True, True

    def reset(self):
        self.reset_done += 1
        return True

    def render(self, mode='human'):
        self.render_done += 1
        return np.random.rand(100, 100, 3) * 255

    def close(self):
        self.close_done += 1


class FakeAgent(AgentInterface):

    @classmethod
    def load(cls, file_name, dire_name=".", device=None):
        pass

    def enable_train(self):
        pass

    def disable_train(self):
        pass

    def save(self, file_name, dire_name="."):
        pass

    def __init__(self, observation_space, action_space, device=None):
        super().__init__(device)
        self.get_action_done = 0
        self.learn_done = 0
        self.episode_finished_done = 0
        self.str_done = 0

    def get_action(self, observation):
        self.get_action_done += 1
        return True

    def learn(self, observation, action, reward, next_observation, done) -> None:
        self.learn_done += 1

    def episode_finished(self) -> None:
        self.episode_finished_done += 1

    def __str__(self):
        self.str_done += 1
        return "FakeAgent"


class FakeLogger(Logger):
    def __init__(self):
        super().__init__()
        self.add_steps_call = 0
        self.add_episode_call = 0
        self.end_episode_call = 0
        self.evaluate_call = 0

    def add_steps(self, steps):
        self.add_steps_call += 1

    def add_episode(self, episode):
        self.add_episode_call += 1

    def end_episode(self):
        self.end_episode_call += 1

    def evaluate(self):
        self.evaluate_call += 1


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
    fake_agent = FakeAgent(observation_space=None, action_space=None)

    trainer = Trainer(environment=fake_env, agent=fake_agent)

    logger = FakeLogger()

    assert fake_agent.episode_finished_done == 0
    assert fake_env.reset_done == 0

    trainer.do_episode()
    assert fake_agent.episode_finished_done == 1
    assert fake_env.reset_done == 1
    assert logger.add_steps_call == 0 and logger.add_episode_call == 0 and logger.end_episode_call == 0

    trainer.do_episode(logger=logger)
    assert fake_agent.episode_finished_done == 2
    assert fake_env.reset_done == 2
    assert logger.add_steps_call == 1 and logger.add_episode_call == 0 and logger.end_episode_call == 1


def test_evaluate():
    fake_env = FakeEnv()
    fake_agent = FakeAgent(observation_space=None, action_space=None)

    trainer = Trainer(environment=fake_env, agent=fake_agent)
    logger = FakeLogger()

    assert fake_agent.episode_finished_done == 0
    assert fake_env.reset_done == 0

    trainer.evaluate()
    assert fake_agent.episode_finished_done == 0 and fake_agent.learn_done == 0
    assert fake_env.reset_done == 1
    assert logger.add_steps_call == 0 and logger.add_episode_call == 0 and logger.end_episode_call == 0
    assert logger.evaluate_call == 0

    trainer.evaluate(logger=logger)
    assert fake_agent.episode_finished_done == 0 and fake_agent.learn_done == 0
    assert fake_env.reset_done == 2
    assert logger.add_steps_call == 1 and logger.add_episode_call == 0 and logger.end_episode_call == 0
    assert logger.evaluate_call == 1


def test_do_step():
    fake_env = FakeEnv()
    fake_agent = FakeAgent(observation_space=None, action_space=None)

    trainer = Trainer(environment=fake_env, agent=fake_agent)
    logger = FakeLogger()

    assert fake_agent.get_action_done == 0 and fake_agent.learn_done == 0 and fake_agent.episode_finished_done == 0
    assert fake_env.step_done == 0 and fake_env.reset_done == 0 and fake_env.render_done == 0

    trainer.do_step(observation=None)
    assert fake_agent.get_action_done == 1 and fake_agent.learn_done == 1 and fake_agent.episode_finished_done == 0
    assert fake_env.step_done == 1 and fake_env.reset_done == 0 and fake_env.render_done == 1
    assert logger.add_steps_call == 0 and logger.add_episode_call == 0 and logger.end_episode_call == 0

    trainer.do_step(observation=None, logger=logger)
    assert fake_agent.get_action_done == 2 and fake_agent.learn_done == 2 and fake_agent.episode_finished_done == 0
    assert fake_env.step_done == 2 and fake_env.reset_done == 0 and fake_env.render_done == 2
    assert logger.add_steps_call == 1 and logger.add_episode_call == 0 and logger.end_episode_call == 0

    trainer.do_step(observation=None, render=False)
    assert fake_agent.get_action_done == 3 and fake_agent.learn_done == 3 and fake_agent.episode_finished_done == 0
    assert fake_env.step_done == 3 and fake_env.reset_done == 0 and fake_env.render_done == 2
    assert logger.add_steps_call == 1 and logger.add_episode_call == 0 and logger.end_episode_call == 0

    trainer.do_step(observation=None, learn=False)
    assert fake_agent.get_action_done == 4 and fake_agent.learn_done == 3 and fake_agent.episode_finished_done == 0
    assert fake_env.step_done == 4 and fake_env.reset_done == 0 and fake_env.render_done == 3
    assert logger.add_steps_call == 1 and logger.add_episode_call == 0 and logger.end_episode_call == 0


def test_init_trainer():
    trainer = Trainer(environment=FakeEnv(), agent=FakeAgent)
    assert isinstance(trainer.agent, AgentInterface) and not isinstance(trainer.agent, type(AgentInterface))
    assert isinstance(trainer.environment, gym.Env)
    assert isinstance(trainer.logger, Logger)

    trainer = Trainer(environment=FakeEnv(), agent=FakeAgent(observation_space=None, action_space=None))
    assert isinstance(trainer.agent, AgentInterface) and not isinstance(trainer.agent, type(AgentInterface))
    assert isinstance(trainer.environment, gym.Env)
    assert isinstance(trainer.logger, Logger)

    with pytest.raises(TypeError):
        Trainer(environment="CartPole-v1", agent="random_agent")

    with pytest.raises(ValueError):
        Trainer(environment="CartPole-dzdv1", agent=FakeAgent)

    trainer = Trainer(environment=FakeEnv(), agent=FakeAgent, log_dir="dede")
    assert isinstance(trainer.agent, AgentInterface) and not isinstance(trainer.agent, type(AgentInterface))
    assert isinstance(trainer.environment, gym.Env)
    assert isinstance(trainer.logger, Logger)
    assert trainer.logger.summary_writer.log_dir == "dede"

    rmtree('dede')


def test_trainer_train():
    test_list = [0, 1, 10, 100, 1000]

    for number_episode in test_list:
        fake_env = FakeEnv()
        fake_agent = FakeAgent(observation_space=None, action_space=None)
        trainer = Trainer(environment=fake_env, agent=fake_agent)

        assert fake_agent.get_action_done == 0 and fake_agent.learn_done == 0 and fake_agent.episode_finished_done == 0
        assert fake_env.step_done == 0 and fake_env.reset_done == 0 and fake_env.render_done == 0

        trainer.train(max_episode=number_episode)

        if number_episode > 4:
            eval = 5
        elif number_episode > 0:
            eval = 1
        else:
            eval = 0

        assert fake_agent.get_action_done == number_episode + eval and fake_agent.learn_done == number_episode
        assert fake_agent.episode_finished_done == number_episode
        assert fake_env.step_done == number_episode + eval and fake_env.reset_done == number_episode + eval + 1
        assert fake_env.render_done == number_episode + eval


def test_render():
    fake_env = FakeEnv()
    fake_agent = FakeAgent(observation_space=None, action_space=None)

    trainer = Trainer(environment=fake_env, agent=fake_agent)

    trainer.render()

    assert fake_env.step_done == 0 and fake_env.reset_done == 0 and fake_env.render_done == 1

    from IPython.testing.globalipapp import get_ipython

    get_ipython().run_line_magic('matplotlib', 'inline')
    # init inline
    trainer.render()
    assert fake_env.step_done == 0 and fake_env.reset_done == 0 and fake_env.render_done == 2
    # maj inline
    trainer.render()
    assert fake_env.step_done == 0 and fake_env.reset_done == 0 and fake_env.render_done == 3


def test_close():
    fake_env = FakeEnv()
    fake_agent = FakeAgent(observation_space=None, action_space=None)

    trainer = Trainer(environment=fake_env, agent=fake_agent)

    trainer.close()
    assert fake_env.step_done == 0 and fake_env.reset_done == 0 and fake_env.render_done == 0
    assert fake_env.close_done == 1

    if platform.system() == "Windows":
        trainer.img = plt.imshow(np.random.rand(100, 100, 3) * 255)
    elif platform.system() == "Linux":
        from pyvirtualdisplay import Display
        trainer.dis = Display(visible=0, size=(1400, 900))
        trainer.dis.start()
        trainer.img = plt.imshow(np.random.rand(100, 100, 3) * 255)

    trainer.close()
    assert fake_env.step_done == 0 and fake_env.reset_done == 0 and fake_env.render_done == 0
    assert fake_env.close_done == 2
    assert not hasattr(trainer, 'img')
