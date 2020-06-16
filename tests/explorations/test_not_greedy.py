from torchforce.explorations import NotGreedy


def test_greedy():
    exploration = NotGreedy()

    assert exploration.be_greedy(0) is False


def test__str__():
    explo = NotGreedy()

    assert 'NotGreedy' == explo.__str__()
