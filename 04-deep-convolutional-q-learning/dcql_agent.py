"""
Deep Convolutional Q-Learning (DCQL) Agent Architecture
Author: Rasit Tekin
Description: Convolutional Neural Network agent trained directly on raw stacked pixel frames
to play Atari-style 2D Pong without hand-crafted features.
"""

import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

NBACTIONS = 3
IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4

OBSERVEPERIOD = 2000
GAMMA = 0.975
BATCH_SIZE = 64
ExpReplay_CAPACITY = 2000


class DCQLAgent:
    def __init__(self):
        self.model = self.create_model()
        self.exp_replay = deque(maxlen=ExpReplay_CAPACITY)
        self.steps = 0
        self.epsilon = 1.0

    def create_model(self):
        """Constructs a 3-layer Convolutional Neural Network with Dense decision head."""
        model = Sequential([
            Conv2D(32, kernel_size=4, strides=(2, 2), padding="same", input_shape=(IMGHEIGHT, IMGWIDTH, IMGHISTORY)),
            Activation("relu"),
            Conv2D(64, kernel_size=4, strides=(2, 2), padding="same"),
            Activation("relu"),
            Conv2D(64, kernel_size=3, strides=(1, 1), padding="same"),
            Activation("relu"),
            Flatten(),
            Dense(512),
            Activation("relu"),
            Dense(NBACTIONS, activation="linear")
        ])
        model.compile(loss="mse", optimizer="adam")
        return model

    def find_best_action(self, state):
        if random.random() < self.epsilon or self.steps < OBSERVEPERIOD:
            return random.randint(0, NBACTIONS - 1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def capture_sample(self, sample):
        self.exp_replay.append(sample)
        self.steps += 1

        # Scheduled exploration decay
        if self.steps <= OBSERVEPERIOD:
            self.epsilon = 1.0
        elif self.steps < 7000:
            self.epsilon = 0.75
        elif self.steps < 14000:
            self.epsilon = 0.50
        elif self.steps < 30000:
            self.epsilon = 0.25
        elif self.steps < 45000:
            self.epsilon = 0.15
        elif self.steps < 70000:
            self.epsilon = 0.10
        else:
            self.epsilon = 0.05

    def process(self):
        """Minibatch experience replay training step."""
        if self.steps <= OBSERVEPERIOD or len(self.exp_replay) < BATCH_SIZE:
            return

        minibatch = random.sample(self.exp_replay, BATCH_SIZE)
        inputs = np.zeros((BATCH_SIZE, IMGHEIGHT, IMGWIDTH, IMGHISTORY))
        targets = np.zeros((BATCH_SIZE, NBACTIONS))

        for i, (state_t, action_t, reward_t, state_t1) in enumerate(minibatch):
            inputs[i:i + 1] = state_t
            targets[i] = self.model.predict(state_t, verbose=0)
            
            if state_t1 is None:
                targets[i, action_t] = reward_t
            else:
                q_next = self.model.predict(state_t1, verbose=0)
                targets[i, action_t] = reward_t + GAMMA * np.max(q_next)

        self.model.fit(inputs, targets, batch_size=BATCH_SIZE, epochs=1, verbose=0)
