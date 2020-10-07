import os
import pytest
import torch
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Dict, Tuple, flatten

from blobrl.agents import DQN
from blobrl.explorations import Greedy, EpsilonGreedy
from blobrl.memories import ExperienceReplay
from blobrl.networks import SimpleNetwork

from tests.agents import TestAgentInterface

TestAgentInterface.__test__ = False


class TestDQN(TestAgentInterface):
    __test__ = True

    agent = DQN
    network = SimpleNetwork

    list_work = [
        [Discrete(3), Discrete(1)],
        [Discrete(3), Discrete(3)],
        [Discrete(10), Discrete(50)],
        [MultiDiscrete([3]), MultiDiscrete([1])],
        [MultiDiscrete([3, 3]), MultiDiscrete([3, 3])],
        [MultiDiscrete([4, 4, 4]), MultiDiscrete([50, 4, 4])],
        [MultiDiscrete([[100, 3], [3, 5]]), MultiDiscrete([[100, 3], [3, 5]])],
        [MultiDiscrete([[[100, 3], [3, 5]], [[100, 3], [3, 5]]]),
         MultiDiscrete([[[100, 3], [3, 5]], [[100, 3], [3, 5]]])]
    ]
    list_fail = [
        [None, None],
        ["dedrfe", "qdzq"],
        [1215.4154, 157.48],
        ["zdzd", (Discrete(1))],
        [Discrete(1), "zdzd"],
        ["zdzd", (1, 4, 7)],
        [(1, 4, 7), "zdzd"],
        [152, 485],
        [MultiBinary(1), MultiBinary(1)],
        [MultiBinary(3), MultiBinary(3)],
        # [MultiBinary([3, 2]), MultiBinary([3, 2])], # Don't work yet because gym don't implemented this
        [Box(low=0, high=10, shape=[1]), Box(low=0, high=10, shape=[1])],
        [Box(low=0, high=10, shape=[2, 2]), Box(low=0, high=10, shape=[2, 2])],
        [Box(low=0, high=10, shape=[2, 2, 2]), Box(low=0, high=10, shape=[2, 2, 2])],

        [Tuple([Discrete(1), MultiDiscrete([1, 1])]), Tuple([Discrete(1), MultiDiscrete([1, 1])])],
        [Dict({"first": Discrete(1), "second": MultiDiscrete([1, 1])}),
         Dict({"first": Discrete(1), "second": MultiDiscrete([1, 1])})],
    ]

    def test_init(self):
        for o, a in self.list_work:
            self.agent(o, a)
            for n in [object(), "dada", 154, 12.1]:
                with pytest.raises(TypeError):
                    self.agent(o, a, neural_network=n)
                with pytest.raises(TypeError):
                    self.agent(o, a, memory=n)
                with pytest.raises(TypeError):
                    self.agent(o, a, loss=n)
                with pytest.raises(TypeError):
                    self.agent(o, a, optimizer=n)
                with pytest.raises(TypeError):
                    self.agent(o, a, greedy_exploration=n)

        for o, a in self.list_fail:
            with pytest.raises(TypeError):
                self.agent(o, a)

    def test_get_action(self):
        for o, a in self.list_work:
            for ge in [Greedy(), EpsilonGreedy(1.)]:
                agent = self.agent(o, a, greedy_exploration=ge)

                for i in range(20):
                    agent.get_action(o.sample())

    def test_learn(self):
        for o, a in self.list_work:
            network = self.network(o, a)
            memory = ExperienceReplay(max_size=5)

            agent = self.agent(observation_space=o, action_space=a, memory=memory, neural_network=network)

            for i in range(20):
                agent.learn(flatten(o, o.sample()), a.sample(), 0, flatten(o, o.sample()), False)

    def test_episode_finished(self):
        for o, a in self.list_work:
            agent = self.agent(observation_space=o, action_space=a)
            agent.episode_finished()

    def test_agent_save_load(self):
        for o, a in self.list_work:
            agent = self.agent(observation_space=o, action_space=a)

            agent.save(file_name="deed.pt")
            agent_l = self.agent.load(file_name="deed.pt")

            assert agent.observation_space == agent_l.observation_space
            assert agent.action_space == agent_l.action_space
            os.remove("deed.pt")

            agent = self.agent(observation_space=o, action_space=a)
            agent.save(file_name="deed.pt", dire_name="./remove/")

            os.remove("./remove/deed.pt")
            os.rmdir("./remove/")

            with pytest.raises(TypeError):
                agent.save(file_name=14548)
            with pytest.raises(TypeError):
                agent.save(file_name="deed.pt", dire_name=14484)

            with pytest.raises(FileNotFoundError):
                self.agent.load(file_name="deed.pt")
            with pytest.raises(FileNotFoundError):
                self.agent.load(file_name="deed.pt", dire_name="/Dede/")

            network = self.network(o, a)

            agent = self.agent(observation_space=o, action_space=a, memory=ExperienceReplay(),
                               neural_network=network,
                               step_train=3, batch_size=12, gamma=0.50,
                               optimizer=torch.optim.Adam(network.parameters()),
                               greedy_exploration=EpsilonGreedy(0.2))

            agent.save(file_name="deed.pt")
            agent_l = self.agent.load(file_name="deed.pt")

            os.remove("deed.pt")

            assert agent.observation_space == agent_l.observation_space
            assert a == agent_l.action_space
            assert isinstance(agent.neural_network, type(agent_l.neural_network))
            for a, b in zip(agent.neural_network.state_dict(), agent_l.neural_network.state_dict()):
                assert a == b
            assert agent.step_train == agent_l.step_train
            assert agent.batch_size == agent_l.batch_size
            assert agent.gamma == agent_l.gamma
            assert isinstance(agent.loss, type(agent_l.loss))
            for a, b in zip(agent.loss.parameters(), agent_l.loss.parameters()):
                assert a == b
            assert isinstance(agent.optimizer, type(agent_l.optimizer))
            for a, b in zip(agent.optimizer.state_dict(), agent_l.optimizer.state_dict()):
                assert a == b
            assert isinstance(agent.greedy_exploration, type(agent_l.greedy_exploration))

    def test_device(self):
        for o, a in self.list_work:
            device = torch.device("cpu")
            assert device == self.agent(o, a, device=device).device

            device = None
            assert torch.device("cpu") == self.agent(o, a, device=device).device

            for device in ["dzeqdzqd", 1512, object(), 151.515]:
                with pytest.raises(TypeError):
                    self.agent(o, a, device=device)

            if torch.cuda.is_available():
                self.agent(o, a, device=torch.device("cuda"))

    def test_dqn_agent_episode_finished(self):
        for o, a in self.list_work:
            network = self.network(o, a)
            memory = ExperienceReplay(max_size=5)

            agent = self.agent(o, a, memory, neural_network=network)
            agent.episode_finished()

    def test_enable_train(self):
        for o, a in self.list_work:
            agent = self.agent(o, a)

            agent.with_exploration = False

            agent.enable_exploration()
            assert agent.with_exploration is True

    def test_disable_train(self):
        for o, a in self.list_work:
            agent = self.agent(o, a)

            agent.disable_exploration()
            assert agent.with_exploration is False

    def test__str__(self):
        raise Exception()
        for o, a in self.list_work:
            agent = self.agent(o, a)

            assert 'DQN-' + str(agent.observation_space) + "-" + str(agent.action_space) + "-" + str(
                agent.neural_network) + "-" + str(agent.memory) + "-" + str(agent.step_train) + "-" + str(
                agent.step) + "-" + str(agent.batch_size) + "-" + str(agent.gamma) + "-" + str(agent.loss) + "-" + str(
                agent.optimizer) + "-" + str(agent.greedy_exploration) == agent.__str__()
