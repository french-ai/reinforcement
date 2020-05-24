from argparse import ArgumentParser

import gym

from torchforce.agents import AgentInterface, AgentRandom


class Trainer:
    def __init__(self, environment, agent):
        self.environment = environment
        if isinstance(agent, str):
            agent = self.arg_to_agent(agent)
        self.agent = agent(gym.make(self.environment).action_space)

    @classmethod
    def arg_to_agent(cls, arg_agent) -> AgentInterface:
        if isinstance(arg_agent, AgentInterface):
            return arg_agent
        if arg_agent == "agent_random":
            return AgentRandom
        raise ValueError("this agent (" + str(arg_agent) + ") is not implemented")

    @classmethod
    def get_environment(cls, arg_env):
        if isinstance(arg_env, str) and arg_env in [env_spec.id for env_spec in gym.envs.registry.all()]:
            return gym.make(arg_env)

        if isinstance(arg_env, gym.Env):
            return arg_env

        raise ValueError("this env (" + str(arg_env) + ") is not supported")

    @classmethod
    def do_episode(cls, env, agent):
        observation = env.reset()
        done = False
        ite = 1
        while not done:
            observation, done = Trainer.do_step(observation=observation, env=env, agent=agent)
            ite += 1
            if done:
                print("Episode finished after {} timesteps".format(ite))
        agent.episode_finished()

    @classmethod
    def do_step(cls, observation, env, agent):
        env.render()
        action = agent.get_action(observation=observation)
        next_observation, reward, done, info = env.step(action)
        agent.learn(observation, action, reward, next_observation)
        return next_observation, done

    def train(self, max_episode=1000):
        env = self.get_environment(self.environment)
        for i_episode in range(max_episode):
            self.do_episode(env, self.agent)
        env.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('agent', type=str, help='name of Agent', nargs='?', const=1, default="agent_random")
    parser.add_argument('env', type=str, help='name of environment', nargs='?', const=1, default="CartPole-v1")
    parser.add_argument('max_episode', type=int, help='number of episode for train', nargs='?', const=1, default=100)
    args = parser.parse_args()

    trainer = Trainer(environment=args.env, agent=args.agent)
    trainer.train(max_episode=args.max_episode)
