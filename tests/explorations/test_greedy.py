from torchforce.explorations import Greedy


def test_greedy():
	exploration = Greedy()

	exploration.be_greedy(0)
