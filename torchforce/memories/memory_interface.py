import abc


class MemoryInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def append(self, observation, action, reward, next_observation, done) -> None:
        pass

    @abc.abstractmethod
    def extend(self, observations, actions, rewards, next_observations, dones) -> None:
        pass

    @abc.abstractmethod
    def sample(self, batch_size):
        pass
