"""
OpenAI Gym Taxi - Baseline Observation & Environment Diagnostics
Author: Rasit Tekin
Description: Demonstrates the state-space encoding, action mappings, and random-policy
baseline execution for the Taxi-v2/v3 environment.
"""

import random
import time

try:
    import gym
except ImportError:
    import gymnasium as gym

try:
    env = gym.make("Taxi-v3").env
except Exception:
    env = gym.make("Taxi-v2").env

print("Observation / State Space Size:", env.observation_space)
print("Action Space Size:", env.action_space)

# Example state encoding: (Taxi Row, Taxi Col, Passenger Loc, Destination Loc)
state = env.encode(3, 1, 2, 3)
print("Encoded State ID for (3, 1, 2, 3):", state)

# Random policy demonstration
total_reward_list = []
for episode in range(5):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    total_reward = 0
    step = 0
    
    while True:
        step += 1
        action = env.action_space.sample()  # Uniform random action
        step_result = env.step(action)
        if len(step_result) == 5:
            state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            state, reward, done, _ = step_result
            
        total_reward += reward
        if done:
            print(f"Episode {episode + 1} finished in {step} steps with Total Reward: {total_reward}")
            total_reward_list.append(total_reward)
            break

print("Random Policy Average Reward:", sum(total_reward_list) / len(total_reward_list))
