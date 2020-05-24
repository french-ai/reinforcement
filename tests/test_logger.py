import pytest

from torchforce import Record


def test_record_init():
    test_fail_values = [None, "deeee", [], {}, object()]
    test_values = [1, 1.0, 0.0, 0, 125, 125.125]

    for value in test_fail_values:
        with pytest.raises(TypeError):
            Record(value=value)

    for value in test_values:
        record = Record(value=value)
        assert record.value == value


def test_avg_records():
    list_records = [[Record(1), Record(1), Record(1), Record(1)],
                    [Record(2), Record(2), Record(2), Record(2)],
                    [Record(1.0), Record(1.0), Record(1.0), Record(1.0)],
                    [Record(2.0), Record(2.0), Record(2.0), Record(2.0)],
                    [Record(1), Record(2), Record(3), Record(4)],
                    [Record(-1), Record(1)],
                    [Record(-10), Record(-15), Record(-20), Record(-15)]]

    list_value = [1, 2, 1.0, 2.0, 2.5, 0, -15]

    for records, value in zip(list_records, list_value):
        assert value == Record.avg_records(records)
