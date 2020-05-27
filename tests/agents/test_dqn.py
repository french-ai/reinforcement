import torch.nn as nn
import torch.nn.functional as F

from gym.spaces import Discrete
from torchforce.agents import DQN
from torchforce.memories import ExperienceReplay
from torchforce.explorations import Greedy

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
	memory = ExperienceReplay(max_size=5)

	agent = DQN(Discrete(4), network, memory)

def test_dqn_agent_instantiation_error_action_space():
	network = Network()
	memory = ExperienceReplay(max_size=5)

	agent = DQN(None, network, memory)

def test_dqn_agent_instantiation_error_neural_network():
	memory = ExperienceReplay(max_size=5)

	agent = DQN(Discrete(4), None, memory)

def test_dqn_agent_instantiation_error_memory():
	network = Network()

	agent = DQN(Discrete(4), network, None)

def test_dqn_agent_instantiation_error_loss():
	network = Network()
	memory = ExperienceReplay(max_size=5)

	agent = DQN(Discrete(4), network, memory, loss="LOSS_ERROR")

def test_dqn_agent_instantiation_error_optimizer():
	network = Network()
	memory = ExperienceReplay(max_size=5)

	agent = DQN(Discrete(4), network, memory, optimizer="OPTIMIZER_ERROR")
	
def test_dqn_agent_instantiation_error_greedy_exploration():
	network = Network()
	memory = ExperienceReplay(max_size=5)

	agent = DQN(Discrete(4), network, memory, greedy_exploration="GREEDY_EXPLORATION_ERROR")

def test_dqn_agent_getaction():
	network = Network()
	memory = ExperienceReplay(max_size=5)

	agent = DQN(Discrete(4), network, memory, greedy_exploration=Greedy())

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

def test_dqn_agent_episode_finished():
	network = Network()
	memory = ExperienceReplay(max_size=5)

	agent = DQN(Discrete(4),network, memory)
	agent.episode_finished()
