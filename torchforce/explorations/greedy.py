from torchforce.explorations import GreedyExplorationInterface


class Greedy(GreedyExplorationInterface):

	def be_greedy(self, step) -> bool:
		return True