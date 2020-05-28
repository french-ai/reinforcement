from random import random

from torchforce.explorations import GreedyExplorationInterface


class EpsilonGreedy(GreedyExplorationInterface):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def be_greedy(self, step):
        return random() > self.epsilon
