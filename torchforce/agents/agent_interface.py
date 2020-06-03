import abc


class AgentInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        """
        pass

    @abc.abstractmethod
    def learn(self, observation, action, reward, next_observation, done) -> None:
        """

        :param observation:
        :param action:
        :param reward:
        :type reward: int, np.int
        :param next_observation:
        :param done:
        """
        pass

    @abc.abstractmethod
    def episode_finished(self) -> None:
        """

        """
        pass

    @abc.abstractmethod
    def save(self, file_name, dire_name="."):
        """

        :param file_name:
        :param dire_name:
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, file_name, dire_name="."):
        """

        :param file_name:
        :param dire_name:
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
