"""
evaluate.py - Performans Kıyaslama ve Simülasyon
Klasik Alarm (Geleneksel) vs. Akıllı RL Ajanı (DQN) karşılaştırması.
"""

import os
import numpy as np
from wake_up_env import SmartWakeUpEnv
from dqn_agent import DQNAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_wake_up_agent.pth")


def run_traditional_alarm(env):
    """
    Geleneksel Alarm Mantığı:
    Hedef saate (0 dk) kadar hiçbir şey yapmaz.
    Hedef saatte ve sonrasında doğrudan son ses alarm çalar.
    """
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        time_to_target = env.time_to_target
        if time_to_target < 0:
            action = env.ACTION_DO_NOTHING
        else:
            action = env.ACTION_ALARM_SOUND

        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    is_success = (info["sleep_stage"] == env.STAGE_AWAKE and info["time_to_target"] <= 5)
    return total_reward, info["snooze_count"], info["sleep_inertia"], is_success


def run_rl_agent(env, agent):
    """
    Eğitilmiş RL Ajanı (DQN):
    Duruma göre ışık, ses ve kademeli eylemleri seçer.
    """
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        state = next_state

    is_success = (info["sleep_stage"] == env.STAGE_AWAKE and info["time_to_target"] <= 5)
    return total_reward, info["snooze_count"], info["sleep_inertia"], is_success


def compare_systems(n_test=100, model_path=None, verbose=True):
    if verbose:
        print("=" * 70)
        print(f"[TEST] KIYASLAMA TESTI: Klasik Alarm vs. RL Akilli Uyanma Asistani ({n_test} Sabah)")
        print("=" * 70)

    env = SmartWakeUpEnv()
    agent = DQNAgent(state_dim=7, action_dim=5)
    resolved_model_path = model_path or DEFAULT_MODEL_PATH
    agent.load(resolved_model_path)

    # 1. Klasik Alarm Testi
    trad_rewards, trad_snoozes, trad_inertias, trad_successes = [], [], [], []
    for _ in range(n_test):
        r, s, i, succ = run_traditional_alarm(env)
        trad_rewards.append(r)
        trad_snoozes.append(s)
        trad_inertias.append(i)
        trad_successes.append(succ)

    # 2. RL Ajanı Testi
    rl_rewards, rl_snoozes, rl_inertias, rl_successes = [], [], [], []
    for _ in range(n_test):
        r, s, i, succ = run_rl_agent(env, agent)
        rl_rewards.append(r)
        rl_snoozes.append(s)
        rl_inertias.append(i)
        rl_successes.append(succ)

    results = {
        "traditional": {
            "success_rate": float(np.mean(trad_successes) * 100),
            "avg_snooze": float(np.mean(trad_snoozes)),
            "avg_inertia": float(np.mean(trad_inertias) * 100),
            "avg_reward": float(np.mean(trad_rewards))
        },
        "rl": {
            "success_rate": float(np.mean(rl_successes) * 100),
            "avg_snooze": float(np.mean(rl_snoozes)),
            "avg_inertia": float(np.mean(rl_inertias) * 100),
            "avg_reward": float(np.mean(rl_rewards))
        }
    }

    if verbose:
        print(f"{'Metrik':<35} | {'Klasik Alarm':<15} | {'RL Akilli Ajan':<15}")
        print("-" * 70)
        print(f"{'Zamaninda Uyanma Basari Orani':<35} | %{results['traditional']['success_rate']:<14.1f} | %{results['rl']['success_rate']:<14.1f}")
        print(f"{'Ortalama Erteleme (Snooze) Sayisi':<35} | {results['traditional']['avg_snooze']:<15.2f} | {results['rl']['avg_snooze']:<15.2f}")
        print(f"{'Ortalama Uyku Sersemligi (Inertia)':<35} | %{results['traditional']['avg_inertia']:<14.1f} | %{results['rl']['avg_inertia']:<14.1f}")
        print(f"{'Ortalama Toplam Odul':<35} | {results['traditional']['avg_reward']:<15.1f} | {results['rl']['avg_reward']:<15.1f}")
        print("=" * 70)

    # --- Örnek Bir Sabahın Zaman Çizelgesi ---
    if verbose:
        print("\n" + "=" * 70)
        print("[SIMULASYON] ORNEK BIR SABAH: RL Ajaninin Karar Zaman Cizelgesi")
        print("=" * 70)

        action_names = [
            "0: Bekle (Uykuyu bolme)",
            "1: Isigi Kademeli Artir (Gun Dogumu)",
            "2: Hafif Melodi / Doga Sesi",
            "3: Standart Alarm Cal",
            "4: Agresif Mod (Maksimum Uyarici)"
        ]

        state, _ = env.reset()
        done = False
        step = 0

        print(f"Baslangic: Uyku suresi={env.total_sleep_hours:.1f} saat | Ilk evre={'Derin' if env.sleep_stage==0 else 'Hafif'} Uyku\n")

        while not done:
            step += 1
            action = agent.select_action(state, evaluate=True)
            act_str = action_names[action]
            next_state, reward, terminated, truncated, info = env.step(action)
            
            env.render()
            print(f"   +---> Secilen Eylem: {act_str} | Anlik Odul: {reward:+.1f}")
            
            state = next_state
            done = terminated or truncated

        print("=" * 70)
        if info["sleep_stage"] == env.STAGE_AWAKE:
            print(f"SONUC: Kullanici {info['time_to_target']:+d}. dakikada YATAKTAN BASARIYLA KALKTI!")
            print(f"Toplam Snooze: {info['snooze_count']} | Kalan Sersemlik: %{int(info['sleep_inertia']*100)}")
        else:
            print("SONUC: Sure asimi, kullanici zamaninda kalkamadi!")
        print("=" * 70)

    return results


if __name__ == "__main__":
    compare_systems(n_test=100)
