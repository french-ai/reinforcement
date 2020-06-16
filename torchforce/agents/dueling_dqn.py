import os
import pickle

import torch
from gym.spaces import flatdim

from torchforce.agents import DoubleDQN
from torchforce.memories import ExperienceReplay
from torchforce.networks import SimpleDuelingNetwork, BaseDuelingNetwork


class DuelingDQN(DoubleDQN):
    """ from 'Dueling Network Architectures for Deep Reinforcement Learning' in https://arxiv.org/abs/1511.06581 """

    def __init__(self, action_space, observation_space, memory=ExperienceReplay(), neural_network=None, step_copy=500,
                 step_train=2, batch_size=32, gamma=0.99, loss=None, optimizer=None, greedy_exploration=None):
        """

        :param action_space:
        :param observation_space:
        :param memory:
        :param neural_network:
        :param step_copy:
        :param step_train:
        :param batch_size:
        :param gamma:
        :param loss:
        :param optimizer:
        :param greedy_exploration:
        """

        if neural_network is None:
            neural_network = SimpleDuelingNetwork(observation_shape=flatdim(observation_space),
                                                  action_shape=flatdim(action_space))

        if not isinstance(neural_network, BaseDuelingNetwork):
            raise TypeError("neural_network need to be instance of torchforce.agents.BaseDuelingNetwork, not :" + str(type(neural_network)))

        super().__init__(action_space, observation_space, memory=memory, neural_network=neural_network,
                         step_copy=step_copy, step_train=step_train, batch_size=batch_size, gamma=gamma, loss=loss,
                         optimizer=optimizer, greedy_exploration=greedy_exploration)

    @classmethod
    def load(cls, file_name, dire_name="."):
        """ load agent form dire_name/file_name

        :param file_name: name of file for load
        :type file_name: string
        :param dire_name: name of directory where we would load it
        :type file_name: string
        """
        dict_save = torch.load(os.path.abspath(os.path.join(dire_name, file_name)))

        neural_network = pickle.loads(dict_save["neural_network_class"])(
            observation_shape=flatdim(pickle.loads(dict_save["observation_space"])),
            action_shape=flatdim(pickle.loads(dict_save["action_space"])))
        neural_network.load_state_dict(dict_save["neural_network"])

        dueling_dqn = DuelingDQN(observation_space=pickle.loads(dict_save["observation_space"]),
                                 action_space=pickle.loads(dict_save["action_space"]),
                                 neural_network=neural_network,
                                 step_train=pickle.loads(dict_save["step_train"]),
                                 batch_size=pickle.loads(dict_save["batch_size"]),
                                 gamma=pickle.loads(dict_save["gamma"]),
                                 loss=pickle.loads(dict_save["loss"]),
                                 optimizer=pickle.loads(dict_save["optimizer"]),
                                 greedy_exploration=pickle.loads(dict_save["greedy_exploration"]))

        dueling_dqn.step_copy = pickle.loads(dict_save["step_copy"])

        return dueling_dqn

    def __str__(self):
        return 'DuelingDQN-' + str(self.observation_space) + "-" + str(self.action_space) + "-" + str(
            self.neural_network) + "-" + str(self.memory) + "-" + str(self.step_train) + "-" + str(
            self.step) + "-" + str(self.batch_size) + "-" + str(self.gamma) + "-" + str(self.loss) + "-" + str(
            self.optimizer) + "-" + str(self.greedy_exploration) + "-" + str(self.step_copy)
