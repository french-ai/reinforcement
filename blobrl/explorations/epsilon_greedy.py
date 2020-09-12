from random import random

from blobrl.explorations import GreedyExplorationInterface


class EpsilonGreedy(GreedyExplorationInterface):

    def __init__(self, epsilon):
        """ Create EpsilonGreedy

        :param epsilon: value for threshold greedy
        :type epsilon: float
        """
        self.epsilon = epsilon

    def be_greedy(self, step):
        """ Return True if random() > on self.epsilon

        :param step: id of step
        :type step: int
        """
        return random() > self.epsilon

    def __str__(self):
        return 'EpsilonGreedy-' + str(self.epsilon)
