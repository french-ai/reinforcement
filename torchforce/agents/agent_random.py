from gym.spaces import Space

from torchforce.agents import AgentInterface


class AgentRandom(AgentInterface):

    def __init__(self, action_space):
        if not isinstance(action_space, Space):
            raise TypeError("action_space need to be instance of gym.spaces.Space, not :" + str(type(action_space)))
        self.action_space = action_space

    def get_action(self, observation):
        return self.action_space.sample()

    def learn(self, observation, action, reward, next_observation) -> None:
        pass

    def episode_finished(self) -> None:
        pass
