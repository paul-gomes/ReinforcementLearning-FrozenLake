import numpy as np
import gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

#parameters
num_episodes = 10000
max_step_per_edisode = 100

learning_rate = 0.1
gamma = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay = 0.001


#q-learning algorithm

rewards_all_episodes = []

for e in range(num_episodes):
    state = env.reset()
    done = False 
    reward_current_episode = 0
    
    for s in range(max_step_per_edisode):
        
        exploration_threshold = random.uniform(0, 1) #generate a random number from 0 to 1
        if exploration_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
            
        new_state, reward, done, info = env.step(action)
        
        #update the q table using bellman equation
        q_table[state, action] = q_table[state,action] * (1 - learning_rate) + learning_rate * (reward + gamma * np.max(q_table[new_state, :]))
        
        state = new_state
        reward_current_episode += reward
        
        if done == True:
            break
        
    # Exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay*e)
    rewards_all_episodes.append(reward_current_episode)
        
        
print(rewards_all_episodes)
# Calculate and print the average reward per thousand episodes
total_rewards = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("*****Average reward per thousand episodes*****\n")
for r in total_rewards:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
    

print("*****Q Table*****\n")
print(q_table)