"""
OpenAI Gym FrozenLake - Tabular Q-Learning Implementation
Author: Rasit Tekin
Description: Solves the stochastic FrozenLake-v0/v1 environment using Q-Learning.
The agent learns to navigate across a slippery frozen grid to reach the goal without falling into holes.
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
    env = gym.make("FrozenLake-v1", is_slippery=True)
except Exception:
    env = gym.make("FrozenLake-v0")

# Initialize Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
gamma = 0.95      # Discount factor
alpha = 0.8       # Learning rate
epsilon = 0.1     # Exploration rate

# Metrics
reward_list = []
episode_number = 75000

print(f"[INFO] Starting FrozenLake Q-Learning Training ({episode_number} episodes)...")

for i in range(1, episode_number):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
        
    reward_count = 0
    
    while True:
        # Epsilon-greedy action choice
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
            
        # Q-Learning update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        next_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = next_value
        
        state = next_state
        reward_count += reward
        
        if done:
            break
            
    if i % 100 == 0:
        reward_list.append(reward_count)
        if i % 10000 == 0:
            avg_recent = np.mean(reward_list[-100:])
            print(f"Episode {i:5d}/{episode_number} | Recent Avg Success Rate: {avg_recent:.2%}")

print("[INFO] Training completed.")
plt.figure(figsize=(10, 4))
plt.plot(reward_list, alpha=0.3, color="#64748B", label="Episode Reward")
window = 100
if len(reward_list) >= window:
    running_avg = np.convolve(reward_list, np.ones(window)/window, mode='valid')
    plt.plot(running_avg, color="#0EA5E9", linewidth=2, label="Moving Average (100-step)")
plt.title("FrozenLake Q-Learning Convergence Curve")
plt.xlabel("Logged Intervals (x100 episodes)")
plt.ylabel("Success Rate")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
