class Trainer:
    def __init__(self, environment, agent, max_episode=1000):
        self.environment = environment
        self.agent = agent
        self.max_episode = max_episode

    def train(self):
        env = self.environment
        for i_episode in range(self.max_episode):
            observation = env.reset()
            done = False
            ite = 1
            while not done:
                env.render()
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(ite))
                ite += 1
        env.close()
