import abc


class AgentInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_action(self, observation):
        """ Return action choice by the agents

        :param observation: stat of environment
        :type observation: gym.Space
        """
        pass

    @abc.abstractmethod
    def learn(self, observation, action, reward, next_observation, done) -> None:
        """ learn from parameters

        :param observation: stat of environment
        :type observation: gym.Space
        :param action: action taken by agent
        :type action: int, float, list
        :param reward: reward win
        :type reward: int, float, np.int, np.float
        :type reward: int, np.int
        :param next_observation:
        :type next_observation: gym.Space
        :param done: if env is finished
        :type done: bool
        """
        pass

    @abc.abstractmethod
    def episode_finished(self) -> None:
        """ Notified agent when episode is done
        """
        pass

    @abc.abstractmethod
    def save(self, file_name, dire_name="."):
        """ Save agent at dire_name/file_name

        :param file_name: name of file for save
        :type file_name: string
        :param dire_name: name of directory where we would save it
        :type file_name: string
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, file_name, dire_name="."):
        """ load agent form dire_name/file_name

        :param file_name: name of file for load
        :type file_name: string
        :param dire_name: name of directory where we would load it
        :type file_name: string
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
