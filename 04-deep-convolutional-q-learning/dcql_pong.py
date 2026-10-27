"""
Pygame Atari Pong Environment for Visual Reinforcement Learning
Author: Rasit Tekin
Description: A headless/visual Pong tennis court simulation with score tracking
and raw RGB surface extraction for Convolutional Neural Network training.
"""

import random
import pygame

FPS = 60
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 420
GAME_HEIGHT = 400

PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60
PADDLE_BUFFER = 15
BALL_WIDTH = 20
BALL_HEIGHT = 20

PADDLE_SPEED = 3
BALL_X_SPEED = 2
BALL_Y_SPEED = 2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class PongGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Atari Pong - DCQL Environment")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.paddle1_y = GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2_y = GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.ball_x = WINDOW_WIDTH / 2
        self.ball_y = random.randint(0, 9) * (WINDOW_HEIGHT - BALL_HEIGHT) / 9
        self.ball_x_dir = random.choice([-1, 1])
        self.ball_y_dir = random.choice([-1, 1])
        self.clock = pygame.time.Clock()
        self.game_score = 0.0

    def _draw_paddle(self, switch, paddle_y):
        if switch == "left":
            paddle = pygame.Rect(PADDLE_BUFFER, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
        else:
            paddle = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, paddle)

    def _draw_ball(self, ball_x, ball_y):
        ball = pygame.Rect(ball_x, ball_y, BALL_WIDTH, BALL_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, ball)

    def _update_paddle(self, switch, action, paddle_y, ball_y):
        dft = 7.5
        if switch == "left":  # Agent paddle
            if action == 1:
                paddle_y -= PADDLE_SPEED * dft
            elif action == 2:
                paddle_y += PADDLE_SPEED * dft
        else:  # Rule-based AI opponent paddle
            if paddle_y + PADDLE_HEIGHT / 2 < ball_y + BALL_HEIGHT / 2:
                paddle_y += PADDLE_SPEED * dft
            elif paddle_y + PADDLE_HEIGHT / 2 > ball_y + BALL_HEIGHT / 2:
                paddle_y -= PADDLE_SPEED * dft

        return max(0, min(GAME_HEIGHT - PADDLE_HEIGHT, paddle_y))

    def _update_ball(self):
        dft = 7.5
        self.ball_x += self.ball_x_dir * BALL_X_SPEED * dft
        self.ball_y += self.ball_y_dir * BALL_Y_SPEED * dft
        score = -0.05  # Living penalty

        # Collision with Agent paddle (Left)
        if (self.ball_x <= (PADDLE_BUFFER + PADDLE_WIDTH)) and \
           ((self.ball_y + BALL_HEIGHT) >= self.paddle1_y) and \
           (self.ball_y <= (self.paddle1_y + PADDLE_HEIGHT)) and (self.ball_x_dir == -1):
            self.ball_x_dir = 1
            score = 10.0
        elif self.ball_x <= 0:  # Missed ball
            self.ball_x_dir = 1
            score = -10.0
            return score

        # Collision with Opponent paddle (Right)
        if (self.ball_x >= (WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER)) and \
           ((self.ball_y + BALL_HEIGHT) >= self.paddle2_y) and \
           (self.ball_y <= (self.paddle2_y + PADDLE_HEIGHT)) and (self.ball_x_dir == 1):
            self.ball_x_dir = -1
        elif self.ball_x >= WINDOW_WIDTH - BALL_WIDTH:
            self.ball_x_dir = -1
            return score

        # Boundary bounces (Top / Bottom)
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_y_dir = 1
        elif self.ball_y >= GAME_HEIGHT - BALL_HEIGHT:
            self.ball_y = GAME_HEIGHT - BALL_HEIGHT
            self.ball_y_dir = -1

        return score

    def play_next_move(self, action):
        self.clock.tick(FPS)
        pygame.event.pump()
        self.screen.fill(BLACK)

        self.paddle1_y = self._update_paddle("left", action, self.paddle1_y, self.ball_y)
        self.paddle2_y = self._update_paddle("right", action, self.paddle2_y, self.ball_y)
        self._draw_paddle("left", self.paddle1_y)
        self._draw_paddle("right", self.paddle2_y)

        score = self._update_ball()
        self._draw_ball(self.ball_x, self.ball_y)

        if score > 0.5 or score < -0.5:
            self.game_score = self.game_score * 0.9 + 0.1 * score

        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        return score, screen_image
