import pytest

from torchforce.explorations import GreedyExplorationInterface


def test_can_t_instantiate_greedy_exploration_interface():
    with pytest.raises(TypeError):
        GreedyExplorationInterface()
