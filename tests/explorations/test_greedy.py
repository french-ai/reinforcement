from torchforce.explorations import Greedy


def test_greedy():
    exploration = Greedy()

    assert exploration.be_greedy(0) is True


def test__str__():
    explo = Greedy()

    assert 'Greedy' == explo.__str__()
