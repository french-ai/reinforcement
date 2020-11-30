import os
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Discrete, MultiDiscrete, flatten

from blobrl.agents import AgentInterface
from blobrl.explorations import GreedyExplorationInterface, EpsilonGreedy
from blobrl.memories import MemoryInterface, ExperienceReplay
from blobrl.networks import SimpleNetwork


class DQN(AgentInterface):

    def enable_exploration(self):
        self.with_exploration = True

    def disable_exploration(self):
        self.with_exploration = False

    def __init__(self, observation_space, action_space, memory=None, network=None, step_train=1, batch_size=32,
                 gamma=1.00, loss=None, optimizer=None, greedy_exploration=None, device=None):
        """


        :param action_space:
        :param observation_space:
        :param memory:
        :param network:
        :param step_train:
        :param batch_size:
        :param gamma:
        :param loss:
        :param optimizer:
        :param greedy_exploration:
        :param device: torch device to run agent
        :type: torch.device
        """

        if not isinstance(action_space, (Discrete, MultiDiscrete)):
            raise TypeError(
                "action_space need to be instance of Discrete or MultiDiscrete, not :" + str(type(action_space)))

        if memory is None:
            memory = ExperienceReplay()

        if not isinstance(memory, MemoryInterface):
            raise TypeError(
                "memory need to be instance of blobrls.memories.MemoryInterface, not :" + str(type(memory)))

        if loss is not None and not isinstance(loss, torch.nn.Module):
            raise TypeError("loss need to be instance of torch.nn.Module, not :" + str(type(loss)))

        if optimizer is not None and not isinstance(optimizer, optim.Optimizer):
            raise TypeError(
                "optimizer need to be instance of torch.optim.Optimizer, not :" + str(type(optimizer)))

        if network is None and optimizer is not None:
            raise TypeError("If network is None, optimizer need to be None not " + str(type(optimizer)))

        if greedy_exploration is not None and not isinstance(greedy_exploration, GreedyExplorationInterface):
            raise TypeError(
                "greedy_exploration need to be instance of blobrls.explorations.GreedyExplorationInterface, not :" + str(
                    type(greedy_exploration)))
        if network is None:
            network = SimpleNetwork(observation_space=observation_space,
                                    action_space=action_space)
        if not isinstance(network, torch.nn.Module):
            raise TypeError("network need to be instance of torch.nn.Module, not :" + str(type(network)))

        super().__init__(observation_space=observation_space, action_space=action_space, device=device)

        self.network = network
        self.network.to(self.device)

        self.memory = memory

        self.step_train = step_train
        self.step = 0
        self.batch_size = batch_size

        self.gamma = gamma

        if loss is None:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = loss

        if optimizer is None:
            self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)
        else:
            self.optimizer = optimizer

        if greedy_exploration is None:
            self.greedy_exploration = EpsilonGreedy(0.1)
        else:
            self.greedy_exploration = greedy_exploration

        self.with_exploration = True

    def get_action(self, observation):
        """ Return action choice by the agents

        :param observation: stat of environment
        :type observation: gym.Space
        """
        if not self.greedy_exploration.be_greedy(self.step) and self.with_exploration:
            return self.action_space.sample()

        observation = torch.tensor([flatten(self.observation_space, observation)], device=self.device).float()

        q_values = self.network.forward(observation)

        def return_values(values):
            if isinstance(values, list):
                return [return_values(v) for v in values]
            else:
                return torch.argmax(values).detach().item()

        return return_values(q_values)

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
        self.memory.append([flatten(self.observation_space, observation)], action, reward,
                           [flatten(self.observation_space, next_observation)], done)
        self.step += 1

        if (self.step % self.step_train) == 0:
            self.train()

    def episode_finished(self) -> None:
        pass

    def train(self):
        """

        """
        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size,
                                                                                      device=self.device)

        next_prediction = self.network.forward(next_observations)

        prediction = self.network.forward(observations)

        if isinstance(self.action_space, Discrete):
            self.apply_loss(next_prediction, prediction, actions, rewards, next_observations, dones,
                            self.action_space.n)
        # find space for one_hot encore action in apply loss
        elif isinstance(self.action_space, MultiDiscrete):
            self.apply_loss(next_prediction, prediction, actions, rewards, next_observations, dones,
                            self.action_space.nvec)
        self.optimizer.step()

    def apply_loss(self, next_prediction, prediction, actions, rewards, next_observations, dones, len_space):
        if isinstance(next_prediction, list):
            [self.apply_loss(n, p, a, rewards, next_observations, dones, c) for n, p, a, c in
             zip(next_prediction, prediction, actions.permute(1, 0, *[i for i in range(2, len(actions.shape))]),
                 len_space)]
        else:

            q = rewards + self.gamma * next_prediction.max(1)[0].detach() * (
                    1 - dones)

            actions_one_hot = F.one_hot(actions.to(torch.int64), num_classes=len_space)
            q_values_predict = prediction * actions_one_hot
            q_predict = torch.max(q_values_predict, dim=1)

            self.optimizer.zero_grad()
            loss = self.loss(q_predict[0], q)
            loss.backward(retain_graph=True)

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
        dict_save["network_class"] = pickle.dumps(type(self.network))
        dict_save["network"] = self.network.cpu().state_dict()
        dict_save["step_train"] = pickle.dumps(self.step_train)
        dict_save["batch_size"] = pickle.dumps(self.batch_size)
        dict_save["gamma"] = pickle.dumps(self.gamma)
        dict_save["loss"] = pickle.dumps(self.loss)
        dict_save["optimizer"] = pickle.dumps(self.optimizer)
        dict_save["greedy_exploration"] = pickle.dumps(self.greedy_exploration)

        torch.save(dict_save, os.path.abspath(os.path.join(dire_name, file_name)))

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

        return DQN(observation_space=pickle.loads(dict_save["observation_space"]),
                   action_space=pickle.loads(dict_save["action_space"]),
                   network=network,
                   step_train=pickle.loads(dict_save["step_train"]),
                   batch_size=pickle.loads(dict_save["batch_size"]),
                   gamma=pickle.loads(dict_save["gamma"]),
                   loss=pickle.loads(dict_save["loss"]),
                   optimizer=pickle.loads(dict_save["optimizer"]),
                   greedy_exploration=pickle.loads(dict_save["greedy_exploration"]),
                   device=device)

    def __str__(self):
        return 'DQN-' + str(self.observation_space) + "-" + str(self.action_space) + "-" + str(
            self.network) + "-" + str(self.memory) + "-" + str(self.step_train) + "-" + str(
            self.step) + "-" + str(self.batch_size) + "-" + str(self.gamma) + "-" + str(self.loss) + "-" + str(
            self.optimizer) + "-" + str(self.greedy_exploration)
