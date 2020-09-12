from blobrl.explorations import GreedyExplorationInterface


class Greedy(GreedyExplorationInterface):

    def be_greedy(self, step) -> bool:
        """ Return True all time

        :param step: id of step
        :type step: int
        """
        return True

    def __str__(self):
        return 'Greedy'
