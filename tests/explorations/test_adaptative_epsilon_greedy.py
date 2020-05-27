from torchforce.explorations import AdaptativeEpsilonGreedy


def test_adaptative_epsilon_greedy_step_min():
	exploration = AdaptativeEpsilonGreedy(0.8, 0.1, 5, 1)

	exploration.be_greedy(0)

def test_adaptative_epsilon_greedy_step():
	exploration = AdaptativeEpsilonGreedy(0.8, 0.1, 5, 1)

	exploration.be_greedy(5)
