from blobrl import Trainer, Record
from blobrl.agents import CategoricalDQN, DQN, DoubleDQN

import gym

if __name__ == "__main__":

    for agent in [CategoricalDQN, DQN, DoubleDQN]:

        env = gym.make("CartPole-v1")
        a = agent(env.observation_space, env.action_space)
        trainer = Trainer(environment=env, agent=agent)

        for i in range(100):

            trainer.train(max_episode=50, render=False, nb_evaluation=0)
            m = max([Record.sum_records(e) for e in trainer.logger.episodes])
            print(agent.__name__, i, m)
            if m > 200:
                break

        print("####### ", agent.__name__, i, m, " #######")
