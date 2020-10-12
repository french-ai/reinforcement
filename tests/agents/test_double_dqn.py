from blobrl.agents import DoubleDQN
from tests.agents import TestDQN
from gym.spaces import flatten
from blobrl.memories import ExperienceReplay


class TestDouble_DQN(TestDQN):
    agent = DoubleDQN

    def test_learn(self):
        for o, a in self.list_work:
            network = self.network(o, a)
            memory = ExperienceReplay(max_size=5)

            agent = self.agent(observation_space=o, action_space=a, memory=memory, step_copy=10, network=network)

            for i in range(20):
                agent.learn(flatten(o, o.sample()), a.sample(), 0, flatten(o, o.sample()), False)

    def test__str__(self):
        for o, a in self.list_work:
            agent = self.agent(o, a)

            assert 'DoubleDQN-' + str(agent.observation_space) + "-" + str(agent.action_space) + "-" + str(
                agent.network) + "-" + str(agent.memory) + "-" + str(agent.step_train) + "-" + str(
                agent.step) + "-" + str(agent.batch_size) + "-" + str(agent.gamma) + "-" + str(agent.loss) + "-" + str(
                agent.optimizer) + "-" + str(agent.greedy_exploration) + "-" + str(agent.step_copy) == agent.__str__()
