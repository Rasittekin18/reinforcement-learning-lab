"""
Gym Taxi Environment - Tabular Q-Learning Implementation
Author: Rasit Tekin
Description: Solves the Taxi environment using the classical Bellman Q-Learning update rule
with an epsilon-greedy action selection policy and performance visualization.
"""

import random
import matplotlib.pyplot as plt
import numpy as np

try:
    import gym
except ImportError:
    import gymnasium as gym

# Initialize environment
try:
    env = gym.make("Taxi-v3").env
except Exception:
    env = gym.make("Taxi-v2").env

# Initialize Q-table (States x Actions)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 0.1     # Exploration rate

# Metrics tracking
reward_list = []
dropouts_list = []

episode_number = 10000

print(f"[INFO] Starting Taxi Q-Learning Training for {episode_number} episodes...")

for i in range(1, episode_number):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
        
    reward_count = 0
    dropouts = 0
    
    while True:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
            
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result
            
        # Bellman Q-Learning Update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        next_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = next_value
        
        state = next_state
        
        if reward == -10:
            dropouts += 1
            
        reward_count += reward
        
        if done:
            break
            
    if i % 100 == 0:
        dropouts_list.append(dropouts)
        reward_list.append(reward_count)
        if i % 1000 == 0:
            print(f"Episode {i:5d}/{episode_number} | Reward: {reward_count:3d} | Wrong Dropouts: {dropouts}")

print("[INFO] Training completed. Visualizing learning metrics...")

# Visualize Training Performance
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(reward_list, color="#0284C7", label="Total Reward")
axs[0].set_title("Reward per 100 Episodes")
axs[0].set_xlabel("Logged Step")
axs[0].set_ylabel("Total Reward")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(dropouts_list, color="#DC2626", label="Penalized Actions")
axs[1].set_title("Wrong Dropouts per 100 Episodes")
axs[1].set_xlabel("Logged Step")
axs[1].set_ylabel("Dropouts Count")
axs[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
