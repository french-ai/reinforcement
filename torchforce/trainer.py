from argparse import ArgumentParser

import gym

from torchforce.agents import AgentInterface, AgentRandom


class Trainer:
    def __init__(self, environment, agent):
        self.environment = environment
        if isinstance(agent, str):
            agent = Trainer.agentarg_to_agent(agent)
        self.agent = agent(gym.make(self.environment).action_space)

    @classmethod
    def agentarg_to_agent(cls, agentargs) -> AgentInterface:
        if agentargs == "agent_random":
            return AgentRandom
        raise NotImplemented("this agent (" + agentargs + " is not implemented")

    def train(self, max_episode=1000):
        env = gym.make(self.environment)
        for i_episode in range(max_episode):
            observation = env.reset()
            done = False
            ite = 1
            while not done:
                env.render()
                action = self.agent.get_action(observation=observation)
                next_observation, reward, done, info = env.step(action)
                self.agent.learn(observation, action, reward, next_observation)
                ite += 1
                observation = next_observation
                if done:
                    print("Episode finished after {} timesteps".format(ite))
            self.agent.episode_finished()
        env.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('agent', type=str, help='name of Agent', nargs='?', const=1, default="agent_random")
    parser.add_argument('env', type=str, help='name of environment', nargs='?', const=1, default="CartPole-v1")
    parser.add_argument('max_episode', type=int, help='number of episode for train', nargs='?', const=1, default=100)
    args = parser.parse_args()

    trainer = Trainer(environment=args.env, agent=args.agent)
    trainer.train(max_episode=args.max_episode)
