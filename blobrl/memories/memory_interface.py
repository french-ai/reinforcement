import abc


class MemoryInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def append(self, observation, action, reward, next_observation, done) -> None:
        """
        Store one couple of value

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
        Store many couple of value

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
        returns *batch_size* sample

        :param device: torch device to run agent
        :type: torch.device
        :param batch_size:
        :type: int
        :return: list<Tensor>
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
