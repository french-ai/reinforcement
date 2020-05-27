from torchforce.explorations import GreedyExplorationInterface
from random import random

class EpsilonGreedy(GreedyExplorationInterface):

	def __init__(self, epsilon):
		self.epsilon = epsilon

	def be_greedy(self, step):
		return random() < self.epsilon
