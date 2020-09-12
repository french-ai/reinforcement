from blobrl.explorations import EpsilonGreedy


def test_epsilon_greedy():
	exploration = EpsilonGreedy(0.4)

	exploration.be_greedy(0)
