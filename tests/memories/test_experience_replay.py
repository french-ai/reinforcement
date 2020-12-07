import torch

from blobrl.memories import ExperienceReplay


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
