import abc

from torchforce.networks import BaseNetwork


class BaseDuelingNetwork(BaseNetwork):
    @abc.abstractmethod
    def __init__(self, observation_shape, action_shape):
        """

        :param observation_shape:
        :param action_shape:
        """
        super().__init__(observation_shape=observation_shape, action_shape=action_shape)

        self.features = None
        self.advantage = None
        self.value = None

    def forward(self, observation):
        x = observation.view(observation.shape[0], -1)
        x = self.features(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()
