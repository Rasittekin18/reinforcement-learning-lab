"""
Pygame Sprite Architecture & Group Collision Demo
Author: Rasit Tekin
Description: Demonstrates object-oriented Sprite implementation with bounded kinematics
and surface rendering for simulation environments.
"""

import random
import pygame

WIDTH = 360
HEIGHT = 360
FPS = 30

WHITE = (255, 255, 255)
BLUE = (14, 165, 233)
GREEN = (34, 197, 94)


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((20, 20))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.y_speed = 5
        self.x_speed = 3
        
    def update(self):
        self.rect.y += self.y_speed
        if self.rect.bottom > HEIGHT - 50:
            self.y_speed = -5
        if self.rect.top < 50:
            self.y_speed = 5


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sprite Architecture Demo")
    clock = pygame.time.Clock()
    
    all_sprites = pygame.sprite.Group()
    player = Player()
    all_sprites.add(player)
    
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        all_sprites.update()
        screen.fill(GREEN)
        all_sprites.draw(screen)
        pygame.display.flip()
        
    pygame.quit()
