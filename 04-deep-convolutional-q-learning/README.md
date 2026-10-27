# 🎾 Module 04: Deep Convolutional Q-Learning (DCQL)

Playing games directly from raw visual input requires extracting spatial features (ball and paddle positions) and temporal dynamics (velocity, trajectories).

## 🖼️ Frame Preprocessing Pipeline

Raw rendering frames contain redundant high-resolution color information. The preprocessing pipeline reduces computational complexity:
1. **Grayscale Conversion (`rgb2gray`):** Reduces color channels from 3 to 1.
2. **Region-of-Interest Cropping:** Removes non-game margins.
3. **Downsampling:** Resizes canvas to $40 \times 40$ resolution.
4. **Intensity Normalization:** Normalizes pixel values ($[0, 255] / 128$).

## ⏱️ Frame Stacking ($40 \times 40 \times 4$)

A single static frame provides no temporal motion cues. By stacking 4 consecutive historical frames along the channel axis, the 2D Convolutional layers learn to calculate directional vectors and speed.

## 🧠 CNN Architecture

```
Input: (40, 40, 4)
  │
  ├── Conv2D(32, 4x4, stride=2) + ReLU
  ├── Conv2D(64, 4x4, stride=2) + ReLU
  ├── Conv2D(64, 3x3, stride=1) + ReLU
  │
  ├── Flatten()
  ├── Dense(512) + ReLU
  │
  └── Dense(3, Linear) -> Q(s, a) for [Idle, Up, Down]
```

## 🚀 Execution

```bash
python 04-deep-convolutional-q-learning/dcql_train.py
```
