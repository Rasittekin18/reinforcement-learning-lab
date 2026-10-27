"""
train.py - Akıllı Uyanma Asistanı DQN Eğitim Scripti
Ajanı 600 sabah simülasyonu boyunca eğitir, ödülleri kaydeder ve grafikleri çizer.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from wake_up_env import SmartWakeUpEnv
from dqn_agent import DQNAgent


def train(episodes=600, render_every=100):
    print("=" * 65)
    print(">> Akilli Uyanma Asistani - DQN Egitimi Baslatiliyor")
    print("=" * 65)

    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    env = SmartWakeUpEnv()
    agent = DQNAgent(
        state_dim=7,
        action_dim=5,
        lr=6e-4,
        gamma=0.96,
        epsilon_start=1.0,
        epsilon_min=0.03,
        epsilon_decay=0.993,
        buffer_capacity=20000,
        batch_size=64,
        target_update_freq=10
    )

    rewards_history = []
    snooze_history = []
    success_history = []
    avg_rewards_window = []

    best_avg_reward = -float("inf")

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward

        agent.decay_epsilon()

        rewards_history.append(episode_reward)
        snooze_history.append(info["snooze_count"])
        # Gerçek Başarı: Tam hedef saat penceresinde (-5 ile +5 dakika arası) yataktan kalkmış olmak
        is_success = (info["sleep_stage"] == env.STAGE_AWAKE and -5 <= info["time_to_target"] <= 5)
        success_history.append(1 if is_success else 0)

        # Son 50 bölümün hareketli ortalaması
        window_size = 50
        if len(rewards_history) >= window_size:
            current_avg_reward = np.mean(rewards_history[-window_size:])
            current_success_rate = np.mean(success_history[-window_size:]) * 100
            current_avg_snooze = np.mean(snooze_history[-window_size:])
        else:
            current_avg_reward = np.mean(rewards_history)
            current_success_rate = np.mean(success_history) * 100
            current_avg_snooze = np.mean(snooze_history)

        avg_rewards_window.append(current_avg_reward)

        # En iyi modeli kaydet
        if episode >= 100 and current_avg_reward > best_avg_reward:
            best_avg_reward = current_avg_reward
            agent.save("models/best_wake_up_agent.pth")

        # Bilgilendirme Çıktısı
        if episode % 25 == 0 or episode == 1:
            print(
                f"Sabah #{episode:03d} | "
                f"Son Odul: {episode_reward:6.1f} | "
                f"Ort. Odul: {current_avg_reward:6.1f} | "
                f"Basari: %{current_success_rate:4.1f} | "
                f"Ort. Snooze: {current_avg_snooze:3.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("\n" + "=" * 65)
    print(f"[OK] Egitim Tamamlandi! En Yuksek Ortalama Odul: {best_avg_reward:.2f}")
    agent.save("models/final_wake_up_agent.pth")
    print("[MODEL] Model agirliklari 'models/best_wake_up_agent.pth' adresine kaydedildi.")

    # --- Performans Grafiklerini Çiz ---
    plt.figure(figsize=(14, 8))

    # 1. Ödül Grafiği
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history, alpha=0.3, color='steelblue', label='Bolum Odulu')
    plt.plot(avg_rewards_window, color='navy', linewidth=2, label='50-Bolum Ortalamasi')
    plt.title("Egitim Surecinde Toplam Odul (Reward)")
    plt.xlabel("Sabah Simulasyonu (Episode)")
    plt.ylabel("Odul")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Başarı Oranı Grafiği
    plt.subplot(2, 2, 2)
    success_rolling = [np.mean(success_history[max(0, i-49):i+1])*100 for i in range(len(success_history))]
    plt.plot(success_rolling, color='forestgreen', linewidth=2)
    plt.title("Zamaninda Uyanma Basari Orani (%)")
    plt.xlabel("Sabah Simulasyonu")
    plt.ylabel("Basari Yuzdesi")
    plt.ylim([0, 105])
    plt.grid(True, alpha=0.3)

    # 3. Erteleme (Snooze) Sayısı Grafiği
    plt.subplot(2, 2, 3)
    snooze_rolling = [np.mean(snooze_history[max(0, i-49):i+1]) for i in range(len(snooze_history))]
    plt.plot(snooze_rolling, color='crimson', linewidth=2)
    plt.title("Ortalama Erteleme (Snooze) Sayisi")
    plt.xlabel("Sabah Simulasyonu")
    plt.ylabel("Snooze Sayisi")
    plt.grid(True, alpha=0.3)

    # 4. Epsilon Azalma Grafiği
    plt.subplot(2, 2, 4)
    eps_history = [max(0.05, 1.0 * (0.992 ** i)) for i in range(episodes)]
    plt.plot(eps_history, color='darkorange', linewidth=2)
    plt.title("Kesif Orani (Epsilon Decay)")
    plt.xlabel("Sabah Simulasyonu")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "plots/training_results.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[GRAFIK] Performans grafigi '{plot_path}' dosyasina kaydedildi.")
    print("=" * 65)


if __name__ == "__main__":
    train(episodes=700)
