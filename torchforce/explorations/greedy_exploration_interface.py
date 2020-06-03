import abc


class GreedyExplorationInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def be_greedy(self, step) -> bool:
        """

        :param step:
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
