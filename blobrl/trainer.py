import sys
import time
from argparse import ArgumentParser

import gym
import matplotlib.pyplot as plt
from IPython import display

from blobrl import Logger, Record
from blobrl.agents import AgentInterface, AgentRandom, DQN, DoubleDQN, CategoricalDQN, DuelingDQN


class Trainer:
    def __init__(self, environment, agent, log_dir="./runs"):
        """

        :param environment:
        :param agent:
        :param log_dir:
        """
        self.environment = self.get_environment(environment)
        if isinstance(agent, type(AgentInterface)):
            action_space = self.get_environment(environment).action_space
            observation_space = self.get_environment(environment).observation_space
            self.agent = agent(observation_space=observation_space, action_space=action_space)
        elif isinstance(agent, AgentInterface):
            import warnings
            warnings.warn("be sure of agent have good input and output dimension")
            self.agent = agent
        else:
            raise TypeError("this type (" + str(type(agent)) + ") is an AgentInterface or instance of AgentInterface")

        self.logger = Logger(log_dir=log_dir)

    @classmethod
    def get_environment(cls, arg_env):
        """

        :param arg_env:
        :return:
        """
        if isinstance(arg_env, str) and arg_env in [env_spec.id for env_spec in gym.envs.registry.all()]:
            return gym.make(arg_env)

        if isinstance(arg_env, gym.Env):
            return arg_env

        raise ValueError("this env (" + str(arg_env) + ") is not supported")

    def do_step(self, observation, learn=True, logger=None, render=True):
        """


        :param observation:
        :param env:
        :param agent:
        :param learn:
        :param logger:
        :param render: if show env render
        :type render: bool
        :param on_notebook: if render is on notebook
        :type on_notebook: bool
        :return:
        """
        if render:
            self.render()
        action = self.agent.get_action(observation=observation)
        next_observation, reward, done, info = self.environment.step(action)
        if learn:
            self.agent.learn(observation, action, reward, next_observation, done)
        if logger:
            logger.add_steps(Record(reward))
        return next_observation, done, reward

    def do_episode(self, logger=None, render=True):
        """

        :param env:
        :param agent:
        :param logger:
        :param render: if show env render
        :type render: bool
        :param on_notebook: if render is on notebook
        :type on_notebook: bool
        """
        self.agent.enable_exploration()
        observation = self.environment.reset()
        done = False
        while not done:
            observation, done, reward = self.do_step(observation=observation,
                                                     learn=True, logger=logger, render=render)
        self.agent.episode_finished()
        if logger:
            logger.end_episode()

    def evaluate(self, logger=None, render=True):
        """

        :param env:
        :param agent:
        :param logger:
        :param render: if show env render
        :type render: bool
        :param on_notebook: if render is on notebook
        :type on_notebook: bool
        """
        self.agent.disable_exploration()
        observation = self.environment.reset()
        done = False
        while not done:
            observation, done, reward = self.do_step(observation=observation,
                                                     learn=False, logger=logger, render=render)
        if logger:
            logger.evaluate()

    def train(self, max_episode=1000, nb_evaluation=4, render=True):
        """

        :param nb_evaluation:
        :param max_episode:
        :param render: if show env render
        :type render: bool
        :param on_notebook: if render is on notebook
        :type on_notebook: bool
        """

        self.environment.reset()
        for i_episode in range(1, max_episode + 1):
            self.do_episode(logger=self.logger, render=render)
            if nb_evaluation != 0:
                if i_episode == 1 or i_episode == max_episode or i_episode % (max_episode // (nb_evaluation - 1)) == 0:
                    self.evaluate(logger=self.logger, render=True)
        self.close()

    def render(self):
        """ Show the environment

        :return:
        """
        if 'inline' in plt.get_backend():
            pub_thread = sys.stdout.pub_thread
            try:
                render = self.environment.render(mode='rgb_array')
                if render is not None:
                    if not hasattr(self, 'img'):
                        plt.figure(figsize=(10, 10))
                        plt.axis('off')
                        self.img = plt.imshow(render)
                    else:
                        self.img.set_data(render)  # just update the data
                    display.display(plt.gcf())
            except Exception as e:
                sys.stdout.pub_thread = pub_thread
                self.environment.render()
                time.sleep(0.5)
            finally:
                display.clear_output(wait=True)
        else:
            self.environment.render()

    def close(self):
        """ Close environment, reset training part for the agent, close render is if on notebook

        :return:
        """
        self.environment.close()
        if hasattr(self, 'img'):
            delattr(self, 'img')


def arg_to_agent(arg_agent) -> AgentInterface:
    """

    :param arg_agent:
    :return:
    """
    if arg_agent == "agent_random":
        return AgentRandom
    if arg_agent == "dqn":
        return DQN
    if arg_agent == "double_dqn":
        return DoubleDQN
    if arg_agent == "categorical_dqn":
        return CategoricalDQN
    if arg_agent == "dueling_dqn":
        return DuelingDQN
    raise ValueError("this agent (" + str(arg_agent) + ") is not implemented")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--agent', type=str, help='name of Agent', nargs='?', const=1, default="agent_random")
    parser.add_argument('--env', type=str, help='name of environment', nargs='?', const=1, default="CartPole-v1")
    parser.add_argument('--max_episode', type=int, help='number of episode to train', nargs='?', const=1, default=100)
    parser.add_argument('--render', type=bool, help='if show render on each step or not', nargs='?', const=1,
                        default=False)
    # parser.add_argument('--train', type=bool, help='if train agent or not', nargs='?', const=1,
    #                    default=True)
    # parser.add_argument('--file_path', type=str, help='path to file for load trained agent')
    args = parser.parse_args()

    trainer = Trainer(environment=args.env, agent=arg_to_agent(args.agent))
    trainer.train(max_episode=args.max_episode, render=args.render)
