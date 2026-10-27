"""
Deep Convolutional Q-Learning (DCQL) Training Pipeline
Author: Rasit Tekin
Description: End-to-end training pipeline connecting Pygame Pong visual output
with 4-frame stacking and CNN agent backpropagation.
"""

import warnings
import numpy as np
import skimage.color
import skimage.exposure
import skimage.transform
from dcql_agent import DCQLAgent
from dcql_pong import PongGame

warnings.filterwarnings("ignore")

TOTAL_TRAIN_TIME = 100000
IMG_HEIGHT = 40
IMG_WIDTH = 40
IMG_HISTORY = 4


def process_game_image(raw_image):
    """Converts 3-channel RGB image to 40x40 normalized grayscale matrix."""
    gray = skimage.color.rgb2gray(raw_image)
    cropped = gray[0:400, 0:400]
    resized = skimage.transform.resize(cropped, (IMG_HEIGHT, IMG_WIDTH))
    rescaled = skimage.exposure.rescale_intensity(resized, out_range=(0, 255))
    return rescaled / 128.0


def train_experiment():
    game = PongGame()
    agent = DCQLAgent()
    
    # Initial warmup step
    best_action = 0
    score, screen_img = game.play_next_move(best_action)
    processed_img = process_game_image(screen_img)
    
    # Stack 4 initial identical frames for history
    game_state = np.stack([processed_img] * IMG_HISTORY, axis=2)
    game_state = game_state.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_HISTORY)
    
    print(f"[INFO] Starting DCQL Training ({TOTAL_TRAIN_TIME} total frames)...")
    
    for frame in range(TOTAL_TRAIN_TIME):
        action = agent.find_best_action(game_state)
        step_score, new_screen = game.play_next_move(action)
        
        new_img = process_game_image(new_screen)
        new_img = new_img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        
        # Shift history: append new frame and drop the oldest frame
        next_state = np.append(new_img, game_state[:, :, :, :3], axis=3)
        
        agent.capture_sample((game_state, action, step_score, next_state))
        agent.process()
        
        game_state = next_state
        
        if frame % 500 == 0:
            print(f"Frame {frame:6d}/{TOTAL_TRAIN_TIME} | Moving Score: {game.game_score:6.2f} | Epsilon: {agent.epsilon:.3f}")


if __name__ == "__main__":
    train_experiment()
