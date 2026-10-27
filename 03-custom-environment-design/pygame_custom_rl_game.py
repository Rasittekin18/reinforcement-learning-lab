"""
Custom Pygame Reinforcement Learning Environment - Falling Objects Evade Game
Author: Rasit Tekin
Description: A custom 2D simulation environment developed in Pygame conforming to
reinforcement learning API standards (step, reset, reward, state-observation).
An autonomous DQN agent learns to dodge incoming falling obstacles in real-time.
"""

import random
from collections import deque
import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Simulation Window Configurations
WIDTH = 360
HEIGHT = 360
FPS = 30

# Color Palettes
WHITE = (255, 255, 255)
BLACK = (15, 23, 42)
RED = (239, 68, 68)
GREEN = (34, 197, 94)
BLUE = (14, 165, 233)


class Player(pygame.sprite.Sprite):
    """The learning agent's avatar moving along the bottom horizontal axis."""
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((20, 20))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.centerx = WIDTH // 2
        self.rect.bottom = HEIGHT - 2
        self.speedx = 0
        
    def update(self, action):
        self.speedx = 0
        keystate = pygame.key.get_pressed()
        
        # 0: Left, 1: Right, 2: Stay
        if keystate[pygame.K_LEFT] or action == 0:
            self.speedx = -5
        elif keystate[pygame.K_RIGHT] or action == 1:
            self.speedx = 5
            
        self.rect.x += self.speedx
        
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0
            
    def get_coordinates(self):
        return (self.rect.x, self.rect.y)


class Enemy(pygame.sprite.Sprite):
    """Falling hazard obstacle spawned at random top horizontal coordinates."""
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((12, 12))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.radius = 6
        pygame.draw.circle(self.image, WHITE, self.rect.center, self.radius)
        self.reset_position()
        self.speedy = 3
        
    def reset_position(self):
        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(2, 10)
        
    def update(self):
        self.rect.y += self.speedy
        if self.rect.top > HEIGHT + 10:
            self.reset_position()
            
    def get_coordinates(self):
        return (self.rect.x, self.rect.y)


class DQLAgent:
    """Deep Q-Network Agent for Custom Pygame Environment."""
    def __init__(self):
        self.state_size = 4  # Relative distances [dx1, dy1, dx2, dy2]
        self.action_size = 3 # 0: Left, 1: Right, 2: Idle
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential([
            Dense(48, input_dim=self.state_size, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state[0]
            targets[i] = self.model.predict(state, verbose=0)[0]
            if done:
                targets[i][action] = reward
            else:
                next_val = np.amax(self.model.predict(next_state, verbose=0)[0])
                targets[i][action] = reward + self.gamma * next_val
                
        self.model.fit(states, targets, epochs=1, verbose=0)
        
    def adaptive_egreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class CustomGameEnv:
    """Gym-style Environment wrapper for the Pygame Falling Obstacles simulation."""
    def __init__(self):
        self.agent = DQLAgent()
        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.reset()
        
    def reset(self):
        self.all_sprites = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        
        self.player = Player()
        self.all_sprites.add(self.player)
        
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.all_sprites.add(self.m1, self.m2)
        self.enemies.add(self.m1, self.m2)
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        return self._get_state()
        
    def _get_state(self):
        p_x, p_y = self.player.get_coordinates()
        m1_x, m1_y = self.m1.get_coordinates()
        m2_x, m2_y = self.m2.get_coordinates()
        return np.array([[p_x - m1_x, p_y - m1_y, p_x - m2_x, p_y - m2_y]])
        
    def step(self, action):
        self.player.update(action)
        self.m1.update()
        self.m2.update()
        
        # Surviving yields positive reward (+2)
        self.reward = 2
        hits = pygame.sprite.spritecollide(self.player, self.enemies, False, pygame.sprite.collide_circle)
        if hits:
            self.reward = -150
            self.done = True
            
        self.total_reward += self.reward
        next_state = self._get_state()
        return next_state, self.reward, self.done


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Custom Pygame RL Environment - Evade Game")
    clock = pygame.time.Clock()
    
    env = CustomGameEnv()
    batch_size = 24
    episodes = 200
    
    print("[INFO] Starting Custom Environment Simulation & Agent Training...")
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        running = True
        
        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    exit()
                    
            action = env.agent.act(state)
            next_state, reward, done = env.step(action)
            
            env.agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            env.agent.replay(batch_size)
            env.agent.adaptive_egreedy()
            
            # Rendering
            screen.fill((16, 185, 129))
            env.all_sprites.draw(screen)
            pygame.display.flip()
            
            if done:
                print(f"Episode {episode:3d} | Total Score: {env.total_reward:4d} | Epsilon: {env.agent.epsilon:.3f}")
                running = False
                
    pygame.quit()
