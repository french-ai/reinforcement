from blobrl.explorations import GreedyExplorationInterface


class NotGreedy(GreedyExplorationInterface):

    def be_greedy(self, step) -> bool:
        """ Return False all time

        :param step: id of step
        :type step: int
        """
        return False

    def __str__(self):
        return 'NotGreedy'
