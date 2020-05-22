import abc


class AgentInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_action(self, observation): pass

    @abc.abstractmethod
    def learn(self, observation, action, reward, next_observation) -> None: pass

    @abc.abstractmethod
    def episode_finished(self) -> None: pass
