from argparse import ArgumentParser

import gym
import torch
from gym.spaces import flatdim
from torch import optim

from torchforce import Trainer
from torchforce.explorations import EpsilonGreedy, AdaptativeEpsilonGreedy
from torchforce.memories import ExperienceReplay
from torchforce.networks import SimpleNetwork
from torchforce.trainer import arg_to_agent

dqn_arg = {
    "memory": [ExperienceReplay],
    "neural_network": [SimpleNetwork],
    "step_train": [1, 2, 10],
    "batch_size": [32, 64, 128],
    "gamma": [0.99, 0.98],
    "loss": [torch.nn.MSELoss()],
    "optimizer": [optim.Adam],
    "lr": [0.1, 0.001, 0.0001],
    "greedy_exploration": [EpsilonGreedy(0.1),
                           EpsilonGreedy(0.2),
                           EpsilonGreedy(0.4),
                           AdaptativeEpsilonGreedy(0.3, 0.1, 50000, 0),
                           AdaptativeEpsilonGreedy(0.6, 0.1, 50000, 0),
                           AdaptativeEpsilonGreedy(0.8, 0.1, 50000, 0)],

}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--agent', type=str, help='name of Agent', nargs='?', const=1, default="agent_random")
    parser.add_argument('--env', type=str, help='name of environment', nargs='?', const=1, default="CartPole-v1")
    parser.add_argument('--max_episode', type=int, help='number of episode for train', nargs='?', const=1, default=500)
    parser.add_argument('--render', type=bool, help='if show render on each step or not', nargs='?', const=1,
                        default=False)
    parser.add_argument('--train', type=bool, help='if train agent or not', nargs='?', const=1,
                        default=True)
    parser.add_argument('--file_path', type=str, help='path to file for load trained agent')
    args = parser.parse_args()

    agent_class = arg_to_agent(args.agent)

    for memory in dqn_arg["memory"]:
        for neural_network_class in dqn_arg["neural_network"]:
            for step_train in dqn_arg["step_train"]:
                for batch_size in dqn_arg["batch_size"]:
                    for gamma in dqn_arg["gamma"]:
                        for loss in dqn_arg["loss"]:
                            for optimizer in dqn_arg["optimizer"]:
                                for lr in dqn_arg["lr"]:
                                    for greedy_exploration in dqn_arg["greedy_exploration"]:
                                        log_dir = str(args.env) + "/" + str(args.agent) + "/" + str(
                                            memory.__name__) + "_" + str(neural_network_class.__name__) + "_" + str(
                                            step_train) + "_" + str(batch_size) + "_" + str(gamma) + "_" + str(
                                            loss) + "_" + str(optimizer.__class__.__name__) + "_" + str(lr) + "_" + str(
                                            greedy_exploration) + "/"

                                        env = gym.make(args.env)

                                        neural_network = neural_network_class(
                                            observation_shape=flatdim(env.observation_space),
                                            action_shape=flatdim(env.action_space)).cpu()

                                        agent = agent_class(observation_space=env.observation_space,
                                                            action_space=env.action_space,
                                                            memory=memory(), neural_network=neural_network,
                                                            step_train=step_train, batch_size=batch_size, gamma=gamma,
                                                            loss=loss,
                                                            optimizer=optimizer(params=neural_network.parameters(),
                                                                                lr=lr),
                                                            greedy_exploration=greedy_exploration)

                                        trainer = Trainer(environment=env, agent=agent,
                                                          log_dir=log_dir)
                                        trainer.train(max_episode=args.max_episode, render=args.render)

                                        agent.save(file_name="save.p", dire_name=log_dir)
