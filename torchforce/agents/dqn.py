import torch

from torchforce.agents import AgentInterface


class DQN(AgentInterface):

	def __init__(self, neural_network=None, memory=None):
		self.neural_network = neural_network
		self.memory = memory

	def get_action(self, observation):
		observation = torch.tensor(observation)
		observation = observation.view(1, -1)

		return torch.argmax(self.neural_network.forward(observation))

	def learn(self, observation, action, reward, next_observation, done) -> None:

		self.memory.append(observation, action, reward, next_observation, done)

	def episode_finished(self) -> None:
		print("TO BE IMPLEMENTED")
