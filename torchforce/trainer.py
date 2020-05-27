from argparse import ArgumentParser

import gym

from torchforce import Logger, Record
from torchforce.agents import AgentInterface, AgentRandom


class Trainer:
    def __init__(self, environment, agent):
        self.environment = environment
        if isinstance(agent, type(AgentInterface)):
            action_space = self.get_environment(environment).action_space
            self.agent = agent(action_space=action_space)
        elif isinstance(agent, AgentInterface):
            import warnings
            warnings.warn("be sure of agent have good input and output dimension")
            self.agent = agent
        else:
            raise TypeError("this type (" + str(type(agent)) + ") is an AgentInterface or instance of AgentInterface")

        self.logger = Logger()

    @classmethod
    def get_environment(cls, arg_env):
        if isinstance(arg_env, str) and arg_env in [env_spec.id for env_spec in gym.envs.registry.all()]:
            return gym.make(arg_env)

        if isinstance(arg_env, gym.Env):
            return arg_env

        raise ValueError("this env (" + str(arg_env) + ") is not supported")

    @classmethod
    def do_step(cls, observation, env, agent, logger=None):
        env.render()
        action = agent.get_action(observation=observation)
        next_observation, reward, done, info = env.step(action)
        agent.learn(observation, action, reward, next_observation, done)
        if logger:
            logger.add_steps(Record(reward))
        return next_observation, done, reward

    @classmethod
    def do_episode(cls, env, agent, logger=None):
        observation = env.reset()
        done = False
        ite = 1
        while not done:
            observation, done, reward = Trainer.do_step(observation=observation, env=env, agent=agent, logger=logger)
            ite += 1
        print("Episode finished after {} timesteps".format(ite))
        agent.episode_finished()
        if logger:
            logger.end_episode()

    def train(self, max_episode=1000):
        env = self.get_environment(self.environment)
        for i_episode in range(max_episode):
            self.do_episode(env=env, agent=self.agent, logger=self.logger)
        env.close()


def arg_to_agent(arg_agent) -> AgentInterface:
    if arg_agent == "agent_random":
        return AgentRandom
    raise ValueError("this agent (" + str(arg_agent) + ") is not implemented")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('agent', type=str, help='name of Agent', nargs='?', const=1, default="agent_random")
    parser.add_argument('env', type=str, help='name of environment', nargs='?', const=1, default="CartPole-v1")
    parser.add_argument('max_episode', type=int, help='number of episode for train', nargs='?', const=1, default=100)
    args = parser.parse_args()

    trainer = Trainer(environment=args.env, agent=arg_to_agent(args.agent))
    trainer.train(max_episode=args.max_episode)
