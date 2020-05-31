import abc


class AgentInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_action(self, observation):
        pass

    @abc.abstractmethod
    def learn(self, observation, action, reward, next_observation, done) -> None:
        pass

    @abc.abstractmethod
    def episode_finished(self) -> None:
        pass

    @abc.abstractmethod
    def save(self, file_name, dire_name="."):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, file_name, dire_name="."):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
