import abc


class MemoryInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def append(self, observation, action, reward, next_observation, done) -> None:
        """

        :param observation:
        :param action:
        :param reward:
        :param next_observation:
        :param done:
        """
        pass

    @abc.abstractmethod
    def extend(self, observations, actions, rewards, next_observations, dones) -> None:
        """

        :param observations:
        :param actions:
        :param rewards:
        :param next_observations:
        :param dones:
        """
        pass

    @abc.abstractmethod
    def sample(self, batch_size, device):
        """

        :param device:
        :param batch_size:
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
