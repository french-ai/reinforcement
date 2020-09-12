import pytest

from blobrl.memories import MemoryInterface


def test_can_t_instantiate_memory_interface():
    with pytest.raises(TypeError):
        MemoryInterface()
