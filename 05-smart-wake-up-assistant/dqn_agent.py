"""
DQNAgent - PyTorch tabanlı Deep Q-Network Ajanı
Akıllı Uyanma Asistanı için Experience Replay, Target Network ve Epsilon-Greedy stratejisi.
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Durum vektörünü alıp her bir eylemin Q değerini tahmin eden derin sinir ağı.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Deneyim Tekrarı (Experience Replay) belleği.
    Ajanın geçmiş deneyimlerinden bağımsız ve rassal örneklem almasını sağlar.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network (DQN) Öğrenme Algoritması
    """
    def __init__(
        self,
        state_dim=7,
        action_dim=5,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-Greedy Parametreleri
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Cihaz (GPU varsa CUDA, yoksa CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Politika Ağı (Policy Net) ve Hedef Ağ (Target Net)
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss (Aykırı değerlere karşı daha kararlı)

        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.train_step_count = 0

    def select_action(self, state, evaluate=False):
        """
        Epsilon-greedy stratejisine göre eylem seçimi.
        evaluate=True ise tamamen greedy (en iyi bilinen eylem) seçilir.
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self):
        """
        Bellman optimizasyon adımı
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a) tahmini
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Hedef: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (patlayan gradyanları engellemek için)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Epsilon değerini kademeli olarak azaltır."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Model ağırlıklarını kaydeder."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Kaydedilmiş ağırlıkları yükler."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
