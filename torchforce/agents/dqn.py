import torch
import torch.optim as optim
import torch.nn.functional as F

from gym.spaces import Discrete
from torchforce.agents import AgentInterface


class DQN(AgentInterface):

	def __init__(self, action_space=None, neural_network=None, memory=None, step_train=2, batch_size=8, gamma=0.99, loss=None, optimizer=None):
		
		if not isinstance(action_space, Discrete):
			raise TypeError("action_space need to be instance of gym.spaces.Space.Discrete, not :" + str(type(action_space)))

		self.action_space = action_space
		self.neural_network = neural_network
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
			self.optimizer = optim.RMSprop(self.neural_network.parameters())
		else:
			self.optimizer = optimizer

	def get_action(self, observation):
		observation = torch.tensor(observation)
		observation = observation.view(1, -1)
		
		return torch.argmax(self.neural_network.forward(observation))

	def learn(self, observation, action, reward, next_observation, done) -> None:

		self.memory.append(observation, action, reward, next_observation, done)
		self.step = (self.step + 1) % self.step_train

		if self.step == 0:
			self.train()

	def episode_finished(self) -> None:
		pass

	def train(self):

		observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)
		
		q = rewards + self.gamma * torch.max(self.neural_network.forward(next_observations), dim=1)[0].detach() * (1 - dones)

		actions_one_hot = F.one_hot(actions.to(torch.int64), num_classes=self.action_space.n)
		q_predict = torch.max(self.neural_network.forward(observations) * actions_one_hot, dim=1)[0]

		self.optimizer.zero_grad()
		loss = self.loss(q_predict, q)
		loss.backward()
		self.optimizer.step()
