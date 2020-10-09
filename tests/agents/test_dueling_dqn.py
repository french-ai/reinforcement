from blobrl.agents import DuelingDQN
from blobrl.networks import SimpleDuelingNetwork
from tests.agents import TestDouble_DQN


class TestDueling_dqn(TestDouble_DQN):
    agent = DuelingDQN
    network = SimpleDuelingNetwork

    def test__str__(self):
        for o, a in self.list_work:
            agent = self.agent(o, a)

            assert 'DuelingDQN-' + str(agent.observation_space) + "-" + str(agent.action_space) + "-" + str(
                agent.network) + "-" + str(agent.memory) + "-" + str(agent.step_train) + "-" + str(
                agent.step) + "-" + str(agent.batch_size) + "-" + str(agent.gamma) + "-" + str(agent.loss) + "-" + str(
                agent.optimizer) + "-" + str(agent.greedy_exploration) + "-" + str(agent.step_copy) == agent.__str__()
