from blobrl.networks import BaseDuelingNetwork
from tests.networks import TestBaseNetwork

class TestBaseDuelingNetwork(TestBaseNetwork):
    __test__ = True

    network = BaseDuelingNetwork
