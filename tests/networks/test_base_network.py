import pytest

from torchforce.networks import BaseNetwork


def test_init_base_network_fail():
    with pytest.raises(TypeError):
        BaseNetwork(None, None)
