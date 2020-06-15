Trainer -- train.py
================

You can start training by using train.py.

# Training
Go to torchforce directory
```bash
cd torchforce
```
start training
```bash
python train.py
```

# Parameters

--agent:
    
 String     
 Default  : agent_random  
 Name of agent listed [*agent_random*, *dqn*, *double_dqn*, *categorical_dqn*]
    
--env:

 String     
 Default : CartPole-v1    
 Name of gym environment listed in [gyms.openai.com](https://gym.openai.com/envs/#classic_control)  
    
--max_episode

 Integer    
 Default : 100  
 Number of episode to train    
 
--render

 Boolean    
 Default : False    
 Show render on each step or not

# Exemples

Start training with DQN on CartPole-v1 with 1000 episodes and show environment
```bash
python train.py --agent dqn --env CartPole-v1 --render 1 --max_episode 1000
```