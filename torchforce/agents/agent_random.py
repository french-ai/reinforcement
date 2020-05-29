import os
import pickle

import torch
from gym.spaces import Space

from torchforce.agents import AgentInterface


class AgentRandom(AgentInterface):

    def __init__(self, observation_space, action_space):
        if not isinstance(action_space, Space):
            raise TypeError("action_space need to be instance of gym.spaces.Space, not :" + str(type(action_space)))
        if not isinstance(observation_space, Space):
            raise TypeError(
                "observation_space need to be instance of gym.spaces.Space, not :" + str(type(observation_space)))
        self.action_space = action_space
        self.observation_space = observation_space

    def get_action(self, observation):
        return self.action_space.sample()

    def learn(self, observation, action, reward, next_observation, done) -> None:
        pass

    def episode_finished(self) -> None:
        pass

    def save(self, file_name, dire_name="."):

        os.makedirs(os.path.abspath(dire_name), exist_ok=True)

        dict_save = dict()
        dict_save["observation_space"] = pickle.dumps(self.observation_space)
        dict_save["action_space"] = pickle.dumps(self.observation_space)

        torch.save(dict_save, os.path.abspath(os.path.join(dire_name, file_name)))

    @classmethod
    def load(cls, file_name, dire_name="."):
        dict_save = torch.load(os.path.abspath(os.path.join(dire_name, file_name)))
        return AgentRandom(observation_space=pickle.loads(dict_save["observation_space"]),
                           action_space=pickle.loads(dict_save["action_space"]))
