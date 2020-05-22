class AgentInterface():
    
	def get_action(self, observation):
		raise NotImplementedError()

	def learn(self, observation, action, reward, next_observation):
		raise NotImplementedError()

	def episod_finished(self):
		raise NotImplementedError()