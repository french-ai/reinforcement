import pytest
from torch.utils.tensorboard import SummaryWriter

from torchforce import Record, Logger


class FakeSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None):
        super().__init__()
        self.log_dir = log_dir
        self.add_scalar_call = []

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.add_scalar_call.append([tag, scalar_value, global_step])


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

    list_fail = ["dsdzs", ["dzdqzdq"], [1548, 1548], 1254]
    for records in list_fail:
        with pytest.raises(TypeError):
            Record.avg_records(records)

    assert 0 == Record.avg_records([])


def test_min_records():
    list_records = [[Record(1), Record(1), Record(1), Record(1)],
                    [Record(2), Record(2), Record(2), Record(2)],
                    [Record(1.0), Record(1.0), Record(1.0), Record(1.0)],
                    [Record(2.0), Record(2.0), Record(2.0), Record(2.0)],
                    [Record(1), Record(2), Record(3), Record(4)],
                    [Record(-1), Record(1)],
                    [Record(-10), Record(-15), Record(-20), Record(-15)]]

    list_value = [1, 2, 1.0, 2.0, 1, -1, -20]

    for records, value in zip(list_records, list_value):
        assert value == Record.min_records(records)

    list_fail = ["dsdzs", ["dzdqzdq"], [1548, 1548], 1254]
    for records in list_fail:
        with pytest.raises(TypeError):
            Record.min_records(records)

    assert 0 == Record.min_records([])


def test_max_records():
    list_records = [[Record(1), Record(1), Record(1), Record(1)],
                    [Record(2), Record(2), Record(2), Record(2)],
                    [Record(1.0), Record(1.0), Record(1.0), Record(1.0)],
                    [Record(2.0), Record(2.0), Record(2.0), Record(2.0)],
                    [Record(1), Record(2), Record(3), Record(4)],
                    [Record(-1), Record(1)],
                    [Record(-10), Record(-15), Record(-20), Record(-15)]]

    list_value = [1, 2, 1.0, 2.0, 4, 1, -10]

    for records, value in zip(list_records, list_value):
        assert value == Record.max_records(records)

    list_fail = ["dsdzs", ["dzdqzdq"], [1548, 1548], 1254]
    for records in list_fail:
        with pytest.raises(TypeError):
            Record.max_records(records)

    assert 0 == Record.max_records([])


def test_sum_records():
    list_records = [[Record(1), Record(1), Record(1), Record(1)],
                    [Record(2), Record(2), Record(2), Record(2)],
                    [Record(1.0), Record(1.0), Record(1.0), Record(1.0)],
                    [Record(2.0), Record(2.0), Record(2.0), Record(2.0)],
                    [Record(1), Record(2), Record(3), Record(4)],
                    [Record(-1), Record(1)],
                    [Record(-10), Record(-15), Record(-20), Record(-15)]]

    list_value = [4, 8, 4.0, 8.0, 10, 0, -60]

    for records, value in zip(list_records, list_value):
        assert value == Record.sum_records(records)

    list_fail = ["dsdzs", ["dzdqzdq"], [1548, 1548], 1254]
    for records in list_fail:
        with pytest.raises(TypeError):
            Record.sum_records(records)

    assert 0 == Record.sum_records([])


def test_logger_init():
    logger = Logger()
    assert not logger.episodes and not logger.current_steps
    assert isinstance(logger.summary_writer, SummaryWriter)

    logger = Logger(log_dir="des")
    assert not logger.episodes and not logger.current_steps
    assert isinstance(logger.summary_writer, SummaryWriter)
    assert logger.summary_writer.log_dir == "des"


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


def test_evaluate():
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
        logger.evaluate()
        assert 0 == len(logger.episodes)


def test_log_episode():
    summary_writer = FakeSummaryWriter()
    list_steps = [[Record(1), Record(1), Record(1), Record(1)],
                  [Record(2), Record(2), Record(2), Record(2)],
                  [Record(1.0), Record(1.0), Record(1.0), Record(1.0)],
                  [Record(2.0), Record(2.0), Record(2.0), Record(2.0)],
                  [Record(1), Record(2), Record(3), Record(4)],
                  [Record(-1), Record(1)],
                  [Record(-10), Record(-15), Record(-20), Record(-15)]]

    for ite, records in enumerate(list_steps):
        Logger.log_episode(summary_writer, records, ite)
        assert (ite + 1) * 4 == len(summary_writer.add_scalar_call)
        assert ite == summary_writer.add_scalar_call[-1][2]


def test_write_log():
    list_steps = [[Record(1), Record(1), Record(1), Record(1)],
                  [Record(2), Record(2), Record(2), Record(2)],
                  [Record(1.0), Record(1.0), Record(1.0), Record(1.0)],
                  [Record(2.0), Record(2.0), Record(2.0), Record(2.0)],
                  [Record(1), Record(2), Record(3), Record(4)],
                  [Record(-1), Record(1)],
                  [Record(-10), Record(-15), Record(-20), Record(-15)]]

    for ite, records in enumerate(list_steps):
        Logger.write_log("./runs", records, ite)
