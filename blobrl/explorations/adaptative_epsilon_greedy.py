from blobrl.explorations import EpsilonGreedy


class AdaptativeEpsilonGreedy(EpsilonGreedy):

    def __init__(self, epsilon_max, epsilon_min, step_max, step_min=0):
        """ Create AdaptativeEpsilonGreedy

        :param epsilon_max: value for start exploration
        :type epsilon_min: float [0.0,1.0], epsilon_max>epsilon_min
        :param epsilon_min: min value exploration
        :type epsilon_min: float [0.0,1.0], epsilon_min<epsilon_max
        :param step_max: step where epsilon start to decrease
        :type step_max: int
        :param step_min: step where greedy return always False
        :type step_min: int
        """
        super().__init__(epsilon_max)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.step_max = step_max
        self.step_min = step_min

    def be_greedy(self, step):
        """ Return greedy

        :param step: id of step
        :type step: int
        """
        if step <= self.step_min:
            return False

        a = (1 / (1 - (self.epsilon_min / self.epsilon_max)) - 1) * self.step_max
        self.epsilon = max((1 - (step / (self.step_max + a))) * self.epsilon_max, self.epsilon_min)
        return super().be_greedy(step)

    def __str__(self):
        return 'AdaptativeEpsilonGreedy-' + str(self.epsilon_max) + '-' + str(self.epsilon_min) + '-' + str(
            self.step_max) + '-' + str(self.step_min)
