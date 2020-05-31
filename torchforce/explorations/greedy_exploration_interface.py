import abc


class GreedyExplorationInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def be_greedy(self, step) -> bool:
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
