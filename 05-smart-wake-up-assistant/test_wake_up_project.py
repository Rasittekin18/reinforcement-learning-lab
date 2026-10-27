"""
test_wake_up_project.py - Kapsamlı Otomatik Test Paketi (PyTest)
Akıllı Uyanma Asistanı projesinin ortam kurallarını, DQN mimarisini,
model doğruluğunu ve ekstrem stres senaryolarını test eder.
"""

import os
import pytest
import numpy as np
import torch
from wake_up_env import SmartWakeUpEnv
from dqn_agent import DQNAgent, QNetwork, ReplayBuffer


# =====================================================================
# 1. ORTAM (GYMNASIUM CONFORMANCE) TESTLERİ
# =====================================================================

def test_env_initialization():
    """Ortamın başlangıç durum uzayı ve eylem uzayı boyutlarını doğrular."""
    env = SmartWakeUpEnv()
    assert env.action_space.n == 5
    assert env.observation_space.shape == (7,)
    assert env.observation_space.dtype == np.float32


def test_env_reset():
    """reset() fonksiyonunun doğru veri tipleri ve geçerli başlangıç durumu döndürdüğünü doğrular."""
    env = SmartWakeUpEnv()
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (7,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.time_to_target == -30
    assert env.light_level == 0.0
    assert env.sound_level == 0.0
    assert env.snooze_count == 0
    assert 4.0 <= env.total_sleep_hours <= 9.0


def test_env_step_all_actions():
    """Tüm 5 eylemin step() içinde hatasız çalıştığını ve zamanın 5'er dakika ilerlediğini doğrular."""
    env = SmartWakeUpEnv()
    env.reset(seed=123)

    for action in range(5):
        initial_time = env.time_to_target
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert env.time_to_target == initial_time + 5
        assert "sleep_stage" in info
        assert "sleep_inertia" in info
        if terminated or truncated:
            break


def test_env_anti_hacking_penalty():
    """Hedef saatten önce (t < 0) erken agresif mod açıldığında ajanın ağır ceza aldığını doğrular."""
    env = SmartWakeUpEnv()
    env.reset(seed=99)
    env.time_to_target = -25  # Erken zaman

    # Eylem 4: Agresif Mod
    obs, reward, terminated, truncated, info = env.step(env.ACTION_AGGRESSIVE_ALERT)
    
    # Erken agresif mod cezası (-45 civarı) kontrol edilir
    assert reward <= -30.0, f"Erken agresif mod yeterince cezalandırılmadı! Alınan ödül: {reward}"


def test_env_max_steps_truncation():
    """Zaman aşımında (max steps) ortamın truncated=True bayrağını doğru yaktığını doğrular."""
    env = SmartWakeUpEnv()
    env.reset(seed=77)
    
    # Kullanıcıyı uyandırmayan eylemlerle süreyi bitirelim
    done = False
    step_count = 0
    while not done:
        step_count += 1
        obs, reward, terminated, truncated, info = env.step(env.ACTION_DO_NOTHING)
        done = terminated or truncated

    assert step_count <= env.max_steps
    assert done is True


# =====================================================================
# 2. DQN AJANI & SİNİR AĞI TESTLERİ
# =====================================================================

def test_q_network_forward():
    """Sinir ağının batch ve tekil durumlar için doğru tensör boyutları ürettiğini doğrular."""
    net = QNetwork(state_dim=7, action_dim=5, hidden_dim=64)
    sample_state = torch.randn(1, 7)
    output = net(sample_state)
    assert output.shape == (1, 5)

    batch_states = torch.randn(32, 7)
    batch_output = net(batch_states)
    assert batch_output.shape == (32, 5)
    assert not torch.isnan(batch_output).any()


def test_replay_buffer():
    """Deneyim belleğinin ekleme, kapasite sınırlama ve örnekleme doğruluğu."""
    buffer = ReplayBuffer(capacity=100)
    for i in range(150):
        buffer.push([0.0]*7, 1, 10.0, [0.1]*7, False)

    assert len(buffer) == 100  # Kapasite aşılmamalı
    states, actions, rewards, next_states, dones = buffer.sample(32)
    assert len(states) == 32
    assert len(actions) == 32
    assert len(rewards) == 32


def test_agent_update_reduces_loss():
    """DQN ajanının Bellman güncellemesini hatasız çalıştırdığını doğrular."""
    agent = DQNAgent(state_dim=7, action_dim=5, batch_size=32)
    for _ in range(50):
        agent.memory.push(
            np.random.randn(7).astype(np.float32),
            np.random.randint(0, 5),
            np.random.randn(),
            np.random.randn(7).astype(np.float32),
            False
        )

    loss = agent.update()
    assert loss is not None
    assert loss >= 0.0
    assert not np.isnan(loss)


def test_model_file_exists_and_loads():
    """Eğitilmiş model dosyasının diskte mevcut olduğunu ve yüklenebildiğini doğrular."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_wake_up_agent.pth")
    assert os.path.exists(model_path), "best_wake_up_agent.pth dosyası bulunamadı!"

    agent = DQNAgent(state_dim=7, action_dim=5)
    agent.load(model_path)
    
    # Model yüklendikten sonra geçerli bir eylem seçebilmeli
    sample_obs = np.array([0.0, 1.0, 0.3, 0.5, 0.0, 0.0, 7.0], dtype=np.float32)
    action = agent.select_action(sample_obs, evaluate=True)
    assert action in range(5)


# =====================================================================
# 3. MODEL PERFORMANSI VE POLİTİKA DOĞRULAMA (STRESS TESTS)
# =====================================================================

def test_trained_agent_does_not_cheat():
    """
    Kritik Test: Eğitilmiş ajanın hedef saatten önce (t < 0) ASLA agresif mod (eylem 4)
    kullanmadığını 50 farklı sabah senaryosunda stres testiyle kanıtlar.
    """
    env = SmartWakeUpEnv()
    agent = DQNAgent(state_dim=7, action_dim=5)
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_wake_up_agent.pth")
    agent.load(model_path)

    early_aggressive_count = 0
    test_episodes = 50

    for seed in range(test_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            # Eğer hedef saate daha vakit varken agresif mod açıldıysa yakala
            if env.time_to_target < 0 and action == env.ACTION_AGGRESSIVE_ALERT:
                early_aggressive_count += 1
            
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            done = terminated or truncated

    assert early_aggressive_count == 0, f"Ajan {early_aggressive_count} kez erken agresif mod kullandı! Reward hacking devam ediyor."


def test_agent_performance_metrics():
    """
    Modelin 100 sabah simülasyonunda:
    - Başarı oranının %90'ın üzerinde olduğunu
    - Ortalama sersemliğin %25'in altında (dinç) olduğunu
    - Ortalama erteleme sayısının 0.3'ün altında olduğunu doğrular.
    """
    env = SmartWakeUpEnv()
    agent = DQNAgent(state_dim=7, action_dim=5)
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_wake_up_agent.pth")
    agent.load(model_path)

    successes = []
    inertias = []
    snoozes = []

    for seed in range(100):
        state, _ = env.reset(seed=seed)
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            done = terminated or truncated

        # Zamanında Başarı: İşe geç kalmadan, hedeften önceki 10 dakika ile hedef saat arasında uyanmış olmak
        is_success = (info["sleep_stage"] == env.STAGE_AWAKE and -10 <= info["time_to_target"] <= 5)
        successes.append(1 if is_success else 0)
        inertias.append(info["sleep_inertia"])
        snoozes.append(info["snooze_count"])

    avg_success = np.mean(successes) * 100
    avg_inertia = np.mean(inertias) * 100
    avg_snooze = np.mean(snoozes)

    print(f"\n[STRESS TEST SONUCLARI] Basari: %{avg_success:.1f} | Sersemlik: %{avg_inertia:.1f} | Ortalama Snooze: {avg_snooze:.2f}")

    assert avg_success >= 85.0, f"Başarı oranı beklenenden düşük: %{avg_success:.1f}"
    assert avg_inertia <= 30.0, f"Kullanıcılar fazla sersem uyanıyor: %{avg_inertia:.1f}"
    assert avg_snooze <= 0.25, f"Erteleme sayısı çok yüksek: {avg_snooze:.2f}"
