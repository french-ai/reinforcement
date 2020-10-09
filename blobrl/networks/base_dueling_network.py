import abc

from blobrl.networks import BaseNetwork


class BaseDuelingNetwork(BaseNetwork):
    @abc.abstractmethod
    def __init__(self, network):
        """

        :param network: network when we add Value head
        :type network: BaseNetwork and not BaseDuelingNetwork
        """

        if not isinstance(network, BaseNetwork):
            raise TypeError("network need to be instance of BaseNetwork, not :" + str(type(network)))

        super().__init__(observation_space=network.observation_space, action_space=network.action_space)
        self.network = network
        self.value_outputs = None

    def forward(self, observation):
        x = observation.view(observation.shape[0], -1)
        x = self.network.network(x)

        def map_forward(layers, last_tensor, value_outputs):
            if isinstance(layers, list):
                return [map_forward(layers, last_tensor, value_outputs) for layers in layers]
            advantage = layers(last_tensor)
            value = value_outputs(last_tensor)
            return value + advantage - advantage.mean()

        return map_forward(self.network.outputs, x, self.value_outputs, )
