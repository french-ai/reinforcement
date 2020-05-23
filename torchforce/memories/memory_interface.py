import abc


class MemoryInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def append(self, observation, action, reward, next_observation, done) -> none:
        pass

    @abc.abstractmethod
    def extends(self, observations, actions, rewards, next_observations, done) -> None:
        pass

    @abc.abstractmethod
    def sample(self, buffer_size):
        pass
