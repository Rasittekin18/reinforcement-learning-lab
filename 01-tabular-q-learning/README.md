# 🚕 Module 01: Tabular Q-Learning

Tabular Q-Learning is a model-free, off-policy Temporal Difference (TD) reinforcement learning algorithm that seeks to find the optimal action-selection policy for any finite Markov Decision Process (MDP).

## 📐 Mathematical Formulation

The Q-value update rule is governed by the Bellman Equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Where:
* $Q(s, a)$: Current expected cumulative reward of taking action $a$ in state $s$.
* $\alpha$: Learning rate ($0 < \alpha \le 1$).
* $\gamma$: Discount factor ($0 \le \gamma < 1$).
* $\max_{a'} Q(s', a')$: Highest Q-value achievable from the subsequent state $s'$.
* $\left[ R_{t+1} + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$: **Temporal Difference (TD) Error**.

## 🎮 Included Environments

### 1. OpenAI Gym Taxi (`Taxi-v2` / `Taxi-v3`)
* **State Space:** 500 discrete states ($5 \times 5 \text{ grid} \times 5 \text{ passenger locations} \times 4 \text{ destinations}$).
* **Action Space:** 6 discrete actions (`South`, `North`, `East`, `West`, `Pickup`, `Dropoff`).
* **Reward:** $+20$ for successful passenger delivery, $-10$ for illegal pickup/dropoff, $-1$ step penalty.
* **Script:** [`gym_taxi_qlearning.py`](./gym_taxi_qlearning.py)

### 2. OpenAI Gym FrozenLake (`FrozenLake-v0` / `FrozenLake-v1`)
* **State Space:** 16 discrete tiles ($4 \times 4$ grid).
* **Action Space:** 4 directions (`Left`, `Down`, `Right`, `Up`).
* **Stochastic Dynamics:** Slippery ice creates transition probability where the agent might slip into orthogonal directions.
* **Reward:** $+1$ for reaching the Goal tile (`G`), $0$ for holes (`H`) and regular ice (`F`).
* **Script:** [`frozen_lake_qlearning.py`](./frozen_lake_qlearning.py)

## 🚀 Execution

```bash
python 01-tabular-q-learning/gym_taxi_qlearning.py
python 01-tabular-q-learning/frozen_lake_qlearning.py
```
