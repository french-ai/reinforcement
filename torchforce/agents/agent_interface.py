import abc


class AgentInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_action(self, observation):
        raise NotImplementedError()

    @abc.abstractmethod
    def learn(self, observation, action, reward, next_observation) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def episode_finished(self) -> None:
        raise NotImplementedError()
