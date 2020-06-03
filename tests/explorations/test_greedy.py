from torchforce.explorations import Greedy


def test_greedy():
    exploration = Greedy()

    exploration.be_greedy(0)


def test__str__():
    explo = Greedy()

    assert 'Greedy' == explo.__str__()
