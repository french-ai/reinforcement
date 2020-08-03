import pytest
import torch
import torch.optim as optim
from gym.spaces import Discrete, Box

from torchforce.agents import CategoricalDQN
from torchforce.explorations import Greedy, EpsilonGreedy
from torchforce.memories import ExperienceReplay
from torchforce.networks import C51Network


def test_categorical_dqn_agent_instantiation():
    CategoricalDQN(Discrete(4), Discrete(4))


def test_categorical_dqn_agent_instantiation_error_action_space():
    with pytest.raises(TypeError):
        CategoricalDQN(None, Discrete(1))


def test_categorical_dqn_agent_instantiation_error_observation_space():
    with pytest.raises(TypeError):
        CategoricalDQN(Discrete(1), None)


def test_categorical_dqn_agent_instantiation_error_neural_network():
    with pytest.raises(TypeError):
        CategoricalDQN(Discrete(4), Discrete(4), neural_network=154)


def test_categorical_dqn_agent_instantiation_error_memory():
    with pytest.raises(TypeError):
        CategoricalDQN(Discrete(4), Discrete(4), None)


def test_categorical_dqn_agent_instantiation_error_loss():
    with pytest.raises(TypeError):
        CategoricalDQN(Discrete(4), Discrete(4), loss="LOSS_ERROR")


def test_categorical_dqn_agent_instantiation_error_optimizer():
    with pytest.raises(TypeError):
        CategoricalDQN(Discrete(4), Discrete(4), optimizer="OPTIMIZER_ERROR")


def test_categorical_dqn_agent_instantiation_error_greedy_exploration():
    with pytest.raises(TypeError):
        CategoricalDQN(Discrete(4), Discrete(4), greedy_exploration="GREEDY_EXPLORATION_ERROR")


def test_categorical_dqn_agent_instantiation_custom_optimizer():
    c51 = C51Network((1), (1))

    CategoricalDQN(Discrete(4), Discrete(4), neural_network=c51, optimizer=optim.RMSprop(c51.parameters()))

    with pytest.raises(TypeError):
        CategoricalDQN(Discrete(4), Discrete(4), neural_network=None, optimizer=optim.RMSprop(c51.parameters()))


def test_categorical_dqn_agent_getaction():
    agent = CategoricalDQN(Discrete(4), Box(0, 3, (3,)), greedy_exploration=Greedy())

    observation = [0, 1, 2]

    agent.get_action(observation)


def test_categorical_dqn_agent_getaction_non_greedy():
    agent = CategoricalDQN(Discrete(4), Box(0, 3, (3,)), greedy_exploration=EpsilonGreedy(1.))

    observation = [0, 1, 2]

    agent.get_action(observation)


def test_categorical_dqn_agent_learn():
    memory = ExperienceReplay(max_size=5)

    agent = CategoricalDQN(Discrete(4), Box(1, 10, (4,)), memory)

    obs = [1, 2, 5, 0]
    action = 0
    reward = 0
    next_obs = [5, 9, 4, 0]
    done = False

    obs_s = [obs, obs, obs]
    actions = [1, 2, 3]
    rewards = [-2.2, 5, 4]
    next_obs_s = [next_obs, next_obs, next_obs]
    dones = [False, True, False]

    memory.extend(obs_s, actions, rewards, next_obs_s, dones)

    agent.learn(obs, action, reward, next_obs, done)
    agent.learn(obs, action, reward, next_obs, done)


def test_categorical_dqn_agent_episode_finished():
    agent = CategoricalDQN(Discrete(4), Discrete(4))
    agent.episode_finished()


def test__str__():
    agent = CategoricalDQN(Discrete(4), Box(1, 10, (4,)))

    assert 'CategoricalDQN-' + str(agent.observation_space) + "-" + str(agent.action_space) + "-" + str(
        agent.neural_network) + "-" + str(agent.memory) + "-" + str(agent.step_train) + "-" + str(
        agent.step) + "-" + str(agent.batch_size) + "-" + str(agent.gamma) + "-" + str(agent.loss) + "-" + str(
        agent.optimizer) + "-" + str(agent.greedy_exploration) + "-" + str(agent.num_atoms) + "-" + str(
        agent.r_min) + "-" + str(agent.r_max) + "-" + str(agent.delta_z) + "-" + str(agent.z) == agent.__str__()


def test_device_gpu():
    if torch.cuda.is_available():
        memory = ExperienceReplay(max_size=5)

        agent = CategoricalDQN(Discrete(4), Box(1, 10, (4,)), memory, device=torch.device("cuda"))

        obs = [1, 2, 5, 0]
        action = 0
        reward = 0
        next_obs = [5, 9, 4, 0]
        done = False

        obs_s = [obs, obs, obs]
        actions = [1, 2, 3]
        rewards = [-2.2, 5, 4]
        next_obs_s = [next_obs, next_obs, next_obs]
        dones = [False, True, False]

        memory.extend(obs_s, actions, rewards, next_obs_s, dones)

        agent.learn(obs, action, reward, next_obs, done)
        agent.learn(obs, action, reward, next_obs, done)
