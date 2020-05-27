import abc


class GreedyExplorationInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def be_greedy(self, step) -> bool:
        pass
