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
        return sum([record.value for record in records]) / len(records)


class Logger:
    def __init__(self):
        self.current_steps = []
        self.episodes = []

    def add_steps(self, steps):
        self.current_steps.append(steps)

    def add_episode(self, episode):
        self.episodes.append(episode)

    def end_episode(self):
        self.episodes.append(self.current_steps)
        self.current_steps = []