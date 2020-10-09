from argparse import ArgumentParser

import gym
import torch
from gym.spaces import flatdim
from torch import optim

from blobrl import Trainer
from blobrl.agents import DQN, DoubleDQN, DuelingDQN, CategoricalDQN
from blobrl.explorations import EpsilonGreedy, AdaptativeEpsilonGreedy
from blobrl.memories import ExperienceReplay
from blobrl.networks import SimpleNetwork, SimpleDuelingNetwork, C51Network

memory = [ExperienceReplay]
step_train = [1, 4, 32]
batch_size = [1, 32, 64]
gamma = [0.99]
loss = [torch.nn.MSELoss()]
optimizer = [optim.Adam]
lr = [0.1, 0.001, 0.0001]

greedy_exploration = [EpsilonGreedy(0.1),
                      EpsilonGreedy(0.6),
                      AdaptativeEpsilonGreedy(0.3, 0.1, 50000, 0),
                      AdaptativeEpsilonGreedy(0.8, 0.2, 50000, 0)]

arg_all = [{"agent": {"class": [DQN, DoubleDQN],
                      "param": {"step_train": step_train,
                                "batch_size": batch_size,
                                "gamma": gamma,
                                "greedy_exploration": greedy_exploration
                                }
                      },
            "network": {"class": [SimpleNetwork],
                               "param": {}},
            "optimizer": {"class": optimizer,
                          "param": {"lr": lr}},
            "memory": {"class": memory,
                       "param": {"max_size": [16, 32, 128]}},
            },
           {"agent": {"class": [DuelingDQN],
                      "param": {"step_train": step_train,
                                "batch_size": batch_size,
                                "gamma": gamma,
                                "greedy_exploration": greedy_exploration
                                }
                      },
            "network": {"class": [SimpleDuelingNetwork],
                               "param": {}},
            "optimizer": {"class": optimizer,
                          "param": {"lr": lr}},
            "memory": {"class": memory,
                       "param": {"max_size": [16, 32, 128]}},
            },
           {"agent": {"class": [CategoricalDQN],
                      "param": {"step_train": step_train,
                                "batch_size": batch_size,
                                "gamma": gamma,
                                "greedy_exploration": greedy_exploration
                                }
                      },
            "network": {"class": [C51Network],
                               "param": {}},
            "optimizer": {"class": optimizer,
                          "param": {"lr": lr}},
            "memory": {"class": memory,
                       "param": {"max_size": [16, 32, 128]}},
            }]


def mzip(*iterables):
    iterators = [iter(it) for it in iterables]
    if not iterators:
        return []
    for elem in iterators[0]:
        if len(iterators) == 1:
            yield [elem]
        else:
            for elems in mzip(*iterables[1:]):
                yield [elem, *elems]


def dict_mzip(x):
    if not x.values():
        yield {}
    for values in mzip(*list(x.values())):
        param = {}
        for key, val in zip(x.keys(), values):
            param[key] = val
        yield param


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, help='name of environment', nargs='?', const=1, default="CartPole-v1")
    parser.add_argument('--max_episode', type=int, help='number of episode for train', nargs='?', const=1, default=500)
    parser.add_argument('--render', type=bool, help='if show render on each step or not', nargs='?', default=False)
    args = parser.parse_args()

    env = gym.make(args.env)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")

    for arg in arg_all:
        arg_agent = arg["agent"]
        arg_network = arg["network"]
        arg_optimizer = arg["optimizer"]
        arg_memory = arg["memory"]

        for class_agent in arg_agent["class"]:
            print("###" + class_agent.__name__)
            for class_network in arg_network["class"]:
                print("  ##" + class_network.__name__)
                for class_optimizer in arg_optimizer["class"]:
                    print("    #" + class_optimizer.__name__)

                    for class_memory in arg_memory["class"]:

                        for param_agent in dict_mzip(arg_agent["param"]):

                            for param_network in dict_mzip(arg_network["param"]):

                                for param_optimizer in dict_mzip(arg_optimizer["param"]):

                                    for param_memory in dict_mzip(arg_memory["param"]):
                                        network = class_network(
                                            observation_shape=flatdim(env.observation_space),
                                            action_shape=flatdim(env.action_space),
                                            **param_network)
                                        optimizer = class_optimizer(network.parameters(), **param_optimizer)

                                        memory = class_memory(**param_memory)

                                        agent = class_agent(observation_space=env.observation_space,
                                                            action_space=env.action_space,
                                                            network=network,
                                                            optimizer=optimizer, memory=memory, device=device,
                                                            **param_agent)

                                        log_dir = args.env + "/" + class_agent.__name__ + "/" + "_".join(
                                            [str(x) for x in list(
                                                param_agent.values())]) + "_" + class_network.__name__ + "_" + "_".join(
                                            [str(x) for x in list(
                                                param_network.values())]) + "_" + class_optimizer.__name__ + "_" + "_".join(
                                            [str(x) for x in list(
                                                param_optimizer.values())]) + "_" + class_memory.__name__ + "_" + "_".join(
                                            [str(x) for x in list(param_memory.values())])

                                        trainer = Trainer(environment=env, agent=agent,
                                                          log_dir=log_dir)
                                        trainer.train(max_episode=args.max_episode, render=args.render)

                                        agent.save(file_name="save.p", dire_name=log_dir)
