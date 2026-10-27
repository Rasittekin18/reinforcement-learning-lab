"""
Deep Q-Network (DQN) for CartPole Balancing
Author: Rasit Tekin
Description: Solves the CartPole-v0/v1 continuous-state environment using a Deep Q-Network
equipped with Experience Replay and Adaptive Epsilon-Greedy Exploration.
"""

import random
from collections import deque
import matplotlib.pyplot as plt
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
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Experience Replay Buffer
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        
    def build_model(self):
        """Constructs the Deep Q-Network function approximator."""
        model = Sequential([
            Dense(48, input_dim=self.state_size, activation="tanh"),
            Dense(48, activation="tanh"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model
        
    def remember(self, state, action, reward, next_state, done):
        """Stores transition in experience replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Selects action via epsilon-greedy policy."""
        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
        
    def replay(self, batch_size):
        """Trains the Q-network by sampling a mini-batch from experience buffer."""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[idx] = state[0]
            current_q = self.model.predict(state, verbose=0)[0]
            
            if done:
                target_val = reward
            else:
                next_q = np.amax(self.model.predict(next_state, verbose=0)[0])
                target_val = reward + self.gamma * next_q
                
            current_q[action] = target_val
            targets[idx] = current_q
            
        self.model.fit(states, targets, epochs=1, verbose=0)
        
    def adaptive_egreedy(self):
        """Decays exploration parameter over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    try:
        env = gym.make("CartPole-v1")
    except Exception:
        env = gym.make("CartPole-v0")
        
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQLAgent(state_size, action_size)
    
    batch_size = 32
    episodes = 150
    duration_history = []
    
    print("[INFO] Starting CartPole DQN Training...")
    
    for e in range(1, episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.reshape(state, [1, state_size])
        
        step_time = 0
        while True:
            step_time += 1
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
            agent.adaptive_egreedy()
            
            if done:
                duration_history.append(step_time)
                if e % 10 == 0:
                    print(f"Episode {e:3d}/{episodes} | Steps Balanced: {step_time:3d} | Epsilon: {agent.epsilon:.3f}")
                break

    print("[INFO] CartPole DQN Training completed.")
