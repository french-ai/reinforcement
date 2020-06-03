from torchforce.explorations import GreedyExplorationInterface


class Greedy(GreedyExplorationInterface):

    def be_greedy(self, step) -> bool:
        """

        :param step:
        :return:
        """
        return True

    def __str__(self):
        return 'Greedy'
