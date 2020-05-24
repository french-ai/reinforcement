import pytest

from torchforce import Record, Logger


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


def test_logger_init():
    logger = Logger()
    assert not logger.episodes and not logger.current_steps


def test_add_step():
    logger = Logger()
    list_records = [Record(1.0), Record(1.0), Record(1.0), Record(1.0)]
    for ite, record in enumerate(list_records):
        logger.add_steps(record)
        assert ite + 1 == len(logger.current_steps)
        assert record == logger.current_steps[-1]


def test_add_episode():
    logger = Logger()
    list_episodes = [[Record(1), Record(1), Record(1), Record(1)],
                     [Record(2), Record(2), Record(2), Record(2)],
                     [Record(1.0), Record(1.0), Record(1.0), Record(1.0)],
                     [Record(2.0), Record(2.0), Record(2.0), Record(2.0)],
                     [Record(1), Record(2), Record(3), Record(4)],
                     [Record(-1), Record(1)],
                     [Record(-10), Record(-15), Record(-20), Record(-15)]]
    for ite, episode in enumerate(list_episodes):
        logger.add_episode(episode)
        assert ite + 1 == len(logger.episodes)
        assert episode == logger.episodes[-1]


def test_end_episode():
    logger = Logger()
    list_steps = [[Record(1), Record(1), Record(1), Record(1)],
                  [Record(2), Record(2), Record(2), Record(2)],
                  [Record(1.0), Record(1.0), Record(1.0), Record(1.0)],
                  [Record(2.0), Record(2.0), Record(2.0), Record(2.0)],
                  [Record(1), Record(2), Record(3), Record(4)],
                  [Record(-1), Record(1)],
                  [Record(-10), Record(-15), Record(-20), Record(-15)]]

    for ite, steps in enumerate(list_steps):
        logger.current_steps = steps
        logger.end_episode()
        assert ite + 1 == len(logger.episodes)
        assert steps == logger.episodes[-1]
