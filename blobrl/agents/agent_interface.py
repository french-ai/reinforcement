import abc

import torch

from gym.spaces import Space


class AgentInterface(metaclass=abc.ABCMeta):

    def __init__(self, observation_space, action_space, device):
        """

        :param device: torch device to run agent
        :type: torch.device
        :param observation_space: Space for init observation size
        :type observation_space: gym.Space
        :param device: torch device to run agent
        :type: torch.device
        """

        if not isinstance(action_space, Space):
            raise TypeError("action_space need to be instance of gym.spaces.Space, not :" + str(type(action_space)))
        if not isinstance(observation_space, Space):
            raise TypeError(
                "observation_space need to be instance of gym.spaces.Space, not :" + str(type(observation_space)))
        self.action_space = action_space
        self.observation_space = observation_space

        if device is None:
            device = torch.device("cpu")
        if not isinstance(device, torch.device):
            raise TypeError("device need to be torch.device instance")
        self.device = device

    @abc.abstractmethod
    def get_action(self, observation):
        """ Return action choice by the agents

        :param observation: stat of environment
        :type observation: gym.Space
        """
        pass

    @abc.abstractmethod
    def enable_exploration(self):
        """Enable train capacity

        :return:
        """
        pass

    @abc.abstractmethod
    def disable_exploration(self):
        """disable train capacity

        :return:
        """
        pass

    @abc.abstractmethod
    def learn(self, observation, action, reward, next_observation, done) -> None:
        """ learn from parameters

        :param observation: stat of environment
        :type observation: gym.Space
        :param action: action taken by agent
        :type action: int, float, list
        :param reward: reward win
        :type reward: int, float, np.int, np.float
        :type reward: int, np.int
        :param next_observation:
        :type next_observation: gym.Space
        :param done: if env is finished
        :type done: bool
        """
        pass

    @abc.abstractmethod
    def episode_finished(self) -> None:
        """ Notified agent when episode is done
        """
        pass

    @abc.abstractmethod
    def save(self, file_name, dire_name="."):
        """ Save agent at dire_name/file_name

        :param file_name: name of file for save
        :type file_name: string
        :param dire_name: name of directory where we would save it
        :type file_name: string
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, file_name, dire_name=".", device=None):
        """ load agent form dire_name/file_name

        :param device: torch device to run agent
        :type: torch.device
        :param file_name: name of file for load
        :type file_name: string
        :param dire_name: name of directory where we would load it
        :type file_name: string
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
