from random import random

from torchforce.explorations import GreedyExplorationInterface


class EpsilonGreedy(GreedyExplorationInterface):

    def __init__(self, epsilon):
        """

        :param epsilon:
        """
        self.epsilon = epsilon

    def be_greedy(self, step):
        """

        :param step:
        :return:
        """
        return random() > self.epsilon

    def __str__(self):
        return 'EpsilonGreedy-' + str(self.epsilon)
