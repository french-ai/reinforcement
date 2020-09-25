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
        self.advantage_outputs = None
        self.value_outputs = None

    def forward(self, observation):
        x = observation.view(observation.shape[0], -1)
        x = self.features(x)

        def map_forward(last_tensor, value_outputs):
            def mp(layers):
                if isinstance(layers, list):
                    return [mp(layers) for layers in layers]
                advantage = layers(last_tensor)
                value = value_outputs(last_tensor)
                return value + advantage - advantage.mean()

            return mp

        return list(map(map_forward(x, self.value_outputs), self.advantage_outputs))
