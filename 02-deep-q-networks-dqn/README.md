# 🚀 Module 02: Deep Q-Networks (DQN)

When the state space becomes continuous or high-dimensional, tabular representation suffers from the **Curse of Dimensionality**. Deep Q-Networks (DQN) solve this by using Deep Neural Networks as non-linear function approximators:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

## 🛡️ Stabilization Mechanisms

Standard Q-learning with neural networks is notoriously unstable or divergent due to:
1. **Correlated Data Transitions:** Addressed via **Experience Replay Memory** ($\mathcal{D} = \{(s, a, r, s', d)\}$), where uniform mini-batches are randomly sampled to break temporal correlations.
2. **Moving Target Problem:** Addressed via a separate **Target Network** ($\theta^-$) whose parameters are updated periodically rather than at every step:
   $$\mathcal{L}(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

## 🎮 Included Benchmarks

* **CartPole Balancing (`CartPole-v0` / `CartPole-v1`):** Inverted pendulum balancing on a moving cart. 4 continuous observations, 2 discrete actions.
* **LunarLander Guidance (`LunarLander-v2`):** 8-dimensional state vector (coordinates, velocities, angle, leg ground contacts), 4 discrete rocket thrust actions.

## 🚀 Execution

```bash
python 02-deep-q-networks-dqn/cartpole_dqn.py
python 02-deep-q-networks-dqn/lunar_lander_dqn.py
```
