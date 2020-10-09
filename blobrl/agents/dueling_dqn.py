import os
import pickle

import torch

from blobrl.agents import DoubleDQN
from blobrl.memories import ExperienceReplay
from blobrl.networks import SimpleDuelingNetwork, BaseDuelingNetwork


class DuelingDQN(DoubleDQN):
    """ from 'Dueling Network Architectures for Deep Reinforcement Learning' in https://arxiv.org/abs/1511.06581 """

    def __init__(self, observation_space, action_space, memory=ExperienceReplay(), network=None, step_copy=500,
                 step_train=2, batch_size=32, gamma=0.99, loss=None, optimizer=None, greedy_exploration=None,
                 device=None):
        """

        :param device: torch device to run agent
        :type: torch.device
        :param action_space:
        :param observation_space:
        :param memory:
        :param network:
        :param step_copy:
        :param step_train:
        :param batch_size:
        :param gamma:
        :param loss:
        :param optimizer:
        :param greedy_exploration:
        """

        if network is None:
            network = SimpleDuelingNetwork(observation_space=observation_space,
                                           action_space=action_space)

        if not isinstance(network, BaseDuelingNetwork):
            raise TypeError("network need to be instance of blobrl.agents.BaseDuelingNetwork, not :" + str(
                type(network)))

        super().__init__(observation_space, action_space, memory=memory, network=network,
                         step_copy=step_copy, step_train=step_train, batch_size=batch_size, gamma=gamma, loss=loss,
                         optimizer=optimizer, greedy_exploration=greedy_exploration, device=device)

    @classmethod
    def load(cls, file_name, dire_name=".", device=None):
        """ load agent form dire_name/file_name

        :param device: torch device to run agent
        :type: torch.device
        :param file_name: name of file for load
        :type file_name: string
        :param dire_name: name of directory where we would load it
        :type file_name: string
        """
        dict_save = torch.load(os.path.abspath(os.path.join(dire_name, file_name)))

        network = pickle.loads(dict_save["network_class"])(
            observation_space=pickle.loads(dict_save["observation_space"]),
            action_space=pickle.loads(dict_save["action_space"]))
        network.load_state_dict(dict_save["network"])

        dueling_dqn = DuelingDQN(observation_space=pickle.loads(dict_save["observation_space"]),
                                 action_space=pickle.loads(dict_save["action_space"]),
                                 network=network,
                                 step_train=pickle.loads(dict_save["step_train"]),
                                 batch_size=pickle.loads(dict_save["batch_size"]),
                                 gamma=pickle.loads(dict_save["gamma"]),
                                 loss=pickle.loads(dict_save["loss"]),
                                 optimizer=pickle.loads(dict_save["optimizer"]),
                                 greedy_exploration=pickle.loads(dict_save["greedy_exploration"]),
                                 device=device)

        dueling_dqn.step_copy = pickle.loads(dict_save["step_copy"])

        return dueling_dqn

    def __str__(self):
        return 'DuelingDQN-' + str(self.observation_space) + "-" + str(self.action_space) + "-" + str(
            self.network) + "-" + str(self.memory) + "-" + str(self.step_train) + "-" + str(
            self.step) + "-" + str(self.batch_size) + "-" + str(self.gamma) + "-" + str(self.loss) + "-" + str(
            self.optimizer) + "-" + str(self.greedy_exploration) + "-" + str(self.step_copy)
