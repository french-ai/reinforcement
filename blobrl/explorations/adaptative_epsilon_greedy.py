from blobrl.explorations import EpsilonGreedy


class AdaptativeEpsilonGreedy(EpsilonGreedy):

    def __init__(self, epsilon_max, epsilon_min, gamma=0.9999):
        """ Create AdaptativeEpsilonGreedy

        :param epsilon_max: value for start exploration
        :type epsilon_min: float [0.0,1.0], epsilon_max>epsilon_min
        :param epsilon_min: min value exploration
        :type epsilon_min: float [0.0,1.0], epsilon_min<epsilon_max
        :param gamma: decrease factor for epsilon
        :type gamma: float [0.0,1.0]
        """
        super().__init__(epsilon_max)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.gamma = gamma

    def be_greedy(self, step):
        """ Return greedy

        :param step: id of step
        :type step: int
        """
        self.epsilon = max(self.epsilon * self.gamma, self.epsilon_min)
        return super().be_greedy(step)

    def __str__(self):
        return 'AdaptativeEpsilonGreedy-' + str(self.epsilon_max) + '-' + str(self.epsilon_min) + '-' + str(self.gamma)
