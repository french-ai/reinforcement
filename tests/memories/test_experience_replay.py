import torch
import pytest
from blobrl.memories import ExperienceReplay

list_fail = [-1, -1.0, -100, -58.654, 1.1, 10, 23.154]
list_work = [0, 1, 0.0, 1.0, 0.5, 0.236515, 0.98]


def test_init_():
    for gamma in list_fail:
        with pytest.raises(ValueError):
            ExperienceReplay(max_size=100, gamma=gamma)

    for gamma in list_work:
        ExperienceReplay(max_size=100, gamma=gamma)


def test_experience_replay():
    max_size = 2

    mem = ExperienceReplay(max_size)

    for i in range(10):
        obs = [1, 2, 5]
        action = 0
        reward = 0
        next_obs = [5, 9, 4]
        done = False

        mem.append(obs, action, reward, next_obs, done)

    mem.sample(2, device=torch.device("cpu"))

    for i in range(10):
        obs_s = [obs, obs, obs]
        actions = [1, 2, 3]
        rewards = [-2.2, 5, 4]
        next_obs_s = [next_obs, next_obs, next_obs]
        dones = [False, True, False]

        mem.extend(obs_s, actions, rewards, next_obs_s, dones)

    mem.sample(2, device=torch.device("cpu"))


def test_get_sample():
    max_size = 10

    for gamma in list_work:
        mem = ExperienceReplay(max_size, gamma=gamma)
        for i in range(10):
            mem.buffer.append([i, i, i, i, False])

        for i in range(10):
            assert mem.get_sample(i)[0] == i

        mem.buffer.append([10, 10, 10, 10, True])
        assert mem.get_sample(0)[0] == 1


def test_str_():
    mem = ExperienceReplay(max_size=1000, gamma=0.5)

    assert mem.__str__() == 'ExperienceReplay-1000-0.5'
