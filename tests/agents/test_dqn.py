import torch.nn as nn
import torch.nn.functional as F

from gym.spaces import Discrete
from torchforce.agents import DQN
from torchforce.memories import ExperienceReplay


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.dense = nn.Linear(3, 4)

	def forward(self, x):
		x = self.dense(x)
		x = F.relu(x)
		return x

def test_dqn_agent_instantiation():

	network = Network()
	agent = DQN(Discrete(4), network)

def test_dqn_agent_getaction():

	network = Network()
	agent = DQN(Discrete(4), network)

	observation = [0.0, 0.5, 1.]

	agent.get_action(observation)

def test_dqn_agent_learn():
	network = Network()
	memory = ExperienceReplay(max_size=5)

	agent = DQN(Discrete(4),network, memory)
	
	obs = [1, 2, 5]
	action = 0
	reward = 0
	next_obs = [5, 9, 4]
	done = False

	obs_s = [obs, obs, obs]
	actions = [1, 2, 3]
	rewards = [-2.2, 5, 4]
	next_obs_s = [next_obs, next_obs, next_obs]
	dones = [False, True, False]

	memory.extend(obs_s, actions, rewards, next_obs_s, dones)

	agent.learn(obs, action, reward, next_obs, done)
	agent.learn(obs, action, reward, next_obs, done)
