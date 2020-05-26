import abc


class AgentInterface(metaclass=abc.ABCMeta):

    def __init__(self, action_space):
        self.action_space = action_space

    @abc.abstractmethod
    def get_action(self, observation):
        pass

    @abc.abstractmethod
    def learn(self, observation, action, reward, next_observation) -> None:
        pass

    @abc.abstractmethod
    def episode_finished(self) -> None:
        pass
