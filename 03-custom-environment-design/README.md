# 🛠️ Module 03: Custom Reinforcement Learning Environment Design

Real-world AI deployment rarely comes with pre-packaged Gym environments. This module demonstrates how to architect, develop, and integrate a custom 2D simulation environment from scratch using **Pygame** and object-oriented reinforcement learning abstractions.

## 🧱 Architecture & Design Patterns

1. **The Game Loop (`Tick -> Event -> Update -> Draw`):** Maintains frame-rate synchronization, gathers user/agent interaction, processes coordinate physics, and renders visual states.
2. **Sprite Hierarchies (`pygame.sprite.Sprite`):** Modular abstraction for interactive actors (Player, Obstacles/Enemies) with bounding radii and circle collision verification.
3. **Gym-Style Interface (`reset()`, `step(action)`):** Provides standard reinforcement learning signatures returning `(observation, reward, done)`.

## 🎮 The Evade Game Specification
* **Objective:** The Player agent must slide left or right at the bottom of the canvas to avoid falling hazards.
* **State Vector:** Relative 2D distances between the agent and both obstacles:
  $$S = [\Delta X_1, \Delta Y_1, \Delta X_2, \Delta Y_2]$$
* **Actions:** `0: Left (-5px)`, `1: Right (+5px)`, `2: Idle (0px)`.
* **Reward Structure:** $+2$ for each survived frame, $-150$ for hazard collision.

## 🚀 Execution

```bash
# Run custom environment template demo
python 03-custom-environment-design/environment_template.py

# Run autonomous evade game agent training
python 03-custom-environment-design/pygame_custom_rl_game.py
```
