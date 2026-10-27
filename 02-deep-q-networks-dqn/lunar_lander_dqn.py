"""
Deep Q-Network (DQN) with Target Network for LunarLander-v2
Author: Rasit Tekin
Description: Autonomous lunar module landing using a dual-network DQN architecture
(Policy Network + Target Network) with vectorized experience replay.
"""

import random
from collections import deque
import numpy as np

try:
    import gym
except ImportError:
    import gymnasium as gym

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9993
        self.gamma = 0.99
        self.learning_rate = 0.0001
        
        # Experience replay & Dual Networks
        self.memory = deque(maxlen=4000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model_update()
        
    def build_model(self):
        """Constructs a 2-hidden layer MLP for Q-value estimation."""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model
        
    def target_model_update(self):
        """Copies weights from active training network to target network."""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
        
    def replay(self, batch_size):
        """Vectorized experience replay with target network calculation."""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch], dtype=int)
        rewards = np.array([x[2] for x in minibatch], dtype=float)
        next_states = np.vstack([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch], dtype=bool)
        
        # Current predictions
        target_q = self.model.predict(states, verbose=0)
        # Next state predictions from target network
        next_q_target = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_target[i])
                
        self.model.fit(states, target_q, epochs=1, verbose=0)
        
    def adaptive_egreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQLAgent(state_size, action_size)
    batch_size = 32
    episodes = 500
    
    print("[INFO] Starting LunarLander-v2 DQN Agent Training...")
    
    for e in range(1, episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.reshape(state, [1, state_size])
        
        total_reward = 0
        for step in range(1000):
            action = agent.act(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
                
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            agent.replay(batch_size)
            total_reward += reward
            
            if done:
                agent.target_model_update()
                break
                
        agent.adaptive_egreedy()
        if e % 10 == 0:
            print(f"Episode {e:4d}/{episodes} | Total Reward: {total_reward:6.1f} | Epsilon: {agent.epsilon:.4f}")
