from gym.spaces import Space

from torchforce.agents import AgentInterface


class AgentRandom(AgentInterface):

    def save(self, save_dir="."):
        pass

    @classmethod
    def load(cls, file):
        pass

    def __init__(self, observation_space, action_space):
        if not isinstance(action_space, Space):
            raise TypeError("action_space need to be instance of gym.spaces.Space, not :" + str(type(action_space)))
        if not isinstance(observation_space, Space):
            raise TypeError(
                "observation_space need to be instance of gym.spaces.Space, not :" + str(type(observation_space)))
        self.action_space = action_space

    def get_action(self, observation):
        return self.action_space.sample()

    def learn(self, observation, action, reward, next_observation, done) -> None:
        pass

    def episode_finished(self) -> None:
        pass
