from torch import nn
from blobrl.networks import BaseDuelingNetwork


class SimpleDuelingNetwork(BaseDuelingNetwork):
    def __init__(self, network):
        """

        :param observation_space:
        :param action_space:
        """

        super().__init__(network=network)

        self.value_outputs = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def __str__(self):
        return 'SimpleDuelingNetwork-' + str(self.network)
