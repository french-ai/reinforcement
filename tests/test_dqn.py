import torch.nn as nn
import torch.nn.functional as F
from torchforce.agents import DQN

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.dense = nn.Linear(3, 4)

	def forward(self, x):
		x = self.dense(x)
		x = F.relu(x)
		return x

def dqn_agent():

	network = Network()
	agent = DQN(network)

	observation = [0.0, 0.5, 1.]

	agent.get_action(observation)

