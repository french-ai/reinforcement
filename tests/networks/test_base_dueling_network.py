import pytest

from blobrl.networks import BaseNetwork


def test_init_base_network_fail():
    with pytest.raises(TypeError):
        BaseNetwork(None, None)
