"""
Pygame Reinforcement Learning Environment - Standard Architecture Template
Author: Rasit Tekin
Description: A boilerplate structure showing the canonical Game Loop pattern
(Input Handling, State Updating, Surface Rendering, Clock Ticking) for RL games.
"""

import random
import pygame

WIDTH = 360
HEIGHT = 360
FPS = 30

WHITE = (255, 255, 255)
BLACK = (15, 23, 42)
GREEN = (34, 197, 94)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Custom RL Environment - Loop Template")
clock = pygame.time.Clock()

running = True
while running:
    clock.tick(FPS)
    
    # 1. Event Processing
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    # 2. Physics & State Updates
    
    # 3. Canvas Rendering
    screen.fill(GREEN)
    
    # 4. Double Buffer Flip
    pygame.display.flip()

pygame.quit()
