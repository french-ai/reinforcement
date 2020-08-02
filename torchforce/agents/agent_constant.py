import os
import pickle

import torch
from gym.spaces import Space

from torchforce.agents import AgentInterface


class AgentConstant(AgentInterface):

    def enable_train(self):
        pass

    def disable_train(self):
        pass

    def __init__(self, observation_space, action_space, device=None):
        """ Create AgentConstant

        :param device: torch device to run agent
        :type: torch.device
        :param observation_space: Space for init observation size
        :type observation_space: gym.Space
        :param action_space: Space for init action size
        :type observation_space: gym.Space
        """
        super().__init__(device)
        if not isinstance(action_space, Space):
            raise TypeError("action_space need to be instance of gym.spaces.Space, not :" + str(type(action_space)))
        if not isinstance(observation_space, Space):
            raise TypeError(
                "observation_space need to be instance of gym.spaces.Space, not :" + str(type(observation_space)))
        self.action_space = action_space
        self.observation_space = observation_space

        self.action = self.action_space.sample()

    def get_action(self, observation):
        """ Return action randomly choice in action_space

        :param observation: stat of environment
        :type observation: gym.Space
        """
        return self.action

    def learn(self, observation, action, reward, next_observation, done) -> None:
        """Learn from parameters, do nothink in AgentRandom

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

    def episode_finished(self) -> None:
        """ Notified agent when episode is done, do nothink in AgentRandom
        """
        pass

    def save(self, file_name, dire_name="."):
        """ Save agent at dire_name/file_name

        :param file_name: name of file for save
        :type file_name: string
        :param dire_name: name of directory where we would save it
        :type file_name: string
        """
        os.makedirs(os.path.abspath(dire_name), exist_ok=True)

        dict_save = dict()
        dict_save["observation_space"] = pickle.dumps(self.observation_space)
        dict_save["action_space"] = pickle.dumps(self.action_space)
        dict_save["action"] = pickle.dumps(self.action)

        torch.save(dict_save, os.path.abspath(os.path.join(dire_name, file_name)))

    @classmethod
    def load(cls, file_name, dire_name=".", device=None):
        """ Load agent form dire_name/file_name

        :param device: torch device to run agent
        :type: torch.device
        :param file_name: name of file for load
        :type file_name: string
        :param dire_name: name of directory where we would load it
        :type file_name: string
        """
        dict_save = torch.load(os.path.abspath(os.path.join(dire_name, file_name)))
        agent = AgentConstant(observation_space=pickle.loads(dict_save["observation_space"]),
                              action_space=pickle.loads(dict_save["action_space"]))
        agent.action = pickle.loads(dict_save["action"])
        return agent

    def __str__(self):
        return 'AgentConstant-' + str(self.observation_space) + "-" + str(self.action_space) + "-" + str(self.action)
