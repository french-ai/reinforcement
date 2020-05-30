from torch.utils.tensorboard import SummaryWriter


class Record:
    def __init__(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("value must be int or float not " + str(type(value)))
        self.value = value

    @classmethod
    def avg_records(cls, records):
        if not isinstance(records, list):
            raise TypeError("records must be a list not " + str(records))
        if not records:
            return 0.0
        try:
            return sum([record.value for record in records]) / len(records)
        except AttributeError:
            raise TypeError("records must a list of Record")

    @classmethod
    def max_records(cls, records):
        if not isinstance(records, list):
            raise TypeError("records must be a list not " + str(records))
        if not records:
            return 0.0
        try:
            return max([record.value for record in records])
        except AttributeError:
            raise TypeError("records must a list of Record")

    @classmethod
    def min_records(cls, records):
        if not isinstance(records, list):
            raise TypeError("records must be a list not " + str(records))
        if not records:
            return 0.0
        try:
            return min([record.value for record in records])
        except AttributeError:
            raise TypeError("records must a list of Record")

    @classmethod
    def sum_records(cls, records):
        if not isinstance(records, list):
            raise TypeError("records must be a list not " + str(records))
        if not records:
            return 0.0
        try:
            return sum([record.value for record in records])
        except AttributeError:
            raise TypeError("records must a list of Record")


class Logger:
    def __init__(self, log_dir="./runs"):
        self.current_steps = []
        self.episodes = []
        self.summary_writer = SummaryWriter(log_dir)

    def add_steps(self, steps):
        self.current_steps.append(steps)

    def add_episode(self, episode):
        self.episodes.append(episode)

    def end_episode(self):
        self.log_episode(self.summary_writer, self.current_steps, len(self.episodes))
        self.episodes.append(self.current_steps)
        self.current_steps = []

    @classmethod
    def log_episode(cls, summary_writer, episode, step):
        summary_writer.add_scalar(tag="Reward/max", scalar_value=Record.max_records(episode), global_step=step)
        summary_writer.add_scalar(tag="Reward/min", scalar_value=Record.max_records(episode), global_step=step)
        summary_writer.add_scalar(tag="Reward/avg", scalar_value=Record.avg_records(episode), global_step=step)
        summary_writer.add_scalar(tag="Reward/sum", scalar_value=Record.sum_records(episode), global_step=step)
