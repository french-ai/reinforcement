from torchforce.explorations import EpsilonGreedy


class AdaptativeEpsilonGreedy(EpsilonGreedy):

    def __init__(self, epsilon_max, epsilon_min, step_max, step_min=0):
        super().__init__(epsilon_max)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.step_max = step_max
        self.step_min = step_min

    def be_greedy(self, step):
        if step <= self.step_min:
            return False

        a = (1 / (1 - (self.epsilon_min / self.epsilon_max)) - 1) * self.step_max
        self.epsilon = max((1 - (step / (self.step_max + a))) * self.epsilon_max, self.epsilon_min)
        return super().be_greedy(step)

    def __str__(self):
        return 'AdaptativeEpsilonGreedy-' + str(self.epsilon_max) + '-' + str(self.epsilon_min) + '-' + str(
            self.step_max) + '-' + str(self.step_min)
