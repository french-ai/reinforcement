import abc


class GreedyExplorationInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def be_greedy(self, step) -> bool:
        """ Return True or False if we need to explore or not

        :param step: id of step
        :type step: int
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
