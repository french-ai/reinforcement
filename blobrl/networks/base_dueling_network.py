import abc

from blobrl.networks import BaseNetwork


class BaseDuelingNetwork(BaseNetwork):
    @abc.abstractmethod
    def __init__(self, observation_space, action_space):
        """

        :param observation_space:
        :param action_space:
        """
        super().__init__(observation_space=observation_space, action_space=action_space)

        self.features = None
        self.advantage = None
        self.value = None

    def forward(self, observation):
        x = observation.view(observation.shape[0], -1)
        x = self.features(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()
