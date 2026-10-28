import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, "models", "model_weights.json")

with open(weights_path, "r", encoding="utf-8") as f:
    weights = json.load(f)

weights_json_str = json.dumps(weights)

header = '''# -*- coding: utf-8 -*-
"""
Akilli Uyanma Asistani - Pydroid 3 Mobil Surumu (Tek Dosya, Sifir Bagimlilik)
Bu script, Pydroid 3 uygulamasinda hicbir kutuphane kurmadan (PyTorch vb. gerekmeden)
saf Python ile calisir ve sabahlari seni yapay zeka ile uyandirir.
"""

import time
import datetime
import json

# --- EGITILMIS YAPAY ZEKA AGIRLIKLARI (DQN MODELI) ---
WEIGHTS = json.loads('''

body = repr(weights_json_str) + ''')

w1 = WEIGHTS["net.0.weight"]
b1 = WEIGHTS["net.0.bias"]
w2 = WEIGHTS["net.2.weight"]
b2 = WEIGHTS["net.2.bias"]
w3 = WEIGHTS["net.4.weight"]
b3 = WEIGHTS["net.4.bias"]

def predict(state):
    # Katman 1: ReLU(W1 * x + b1)
    h1 = [max(0.0, sum(w1[i][j] * state[j] for j in range(7)) + b1[i]) for i in range(64)]
    # Katman 2: ReLU(W2 * h1 + b2)
    h2 = [max(0.0, sum(w2[i][j] * h1[j] for j in range(64)) + b2[i]) for i in range(64)]
    # Cikis: W3 * h2 + b3
    q = [sum(w3[i][j] * h2[j] for j in range(64)) + b3[i] for i in range(5)]
    return max(range(5), key=lambda i: q[i])

ACTION_NAMES = [
    "[0] Bekle (Uykuyu bolme)",
    "[1] Isigi Kademeli Artir (Gun Dogumu Similasyonu)",
    "[2] Hafif Melodi / Doga Sesi Cal",
    "[3] Standart Alarm Cal (Uyandirma Vakti)",
    "[4] Agresif Mod (Maksimum Uyarici)"
]

def run_simulation(fast_mode=False):
    step_delay = 3 if fast_mode else 300  # Test icin 3 saniye, gercekte 5 dakika (300 sn)
    time_to_target = -30
    sleep_stage = 0  # 0: Derin, 1: Hafif, 2: Yari Uyanik, 3: Kalkti
    sleep_inertia = 0.5
    light_level = 0.0
    sound_level = 0.0
    snooze_count = 0
    total_sleep_hours = 7.0

    print("\\n" + "="*50)
    print(">> AKILLI UYANMA BASLATILDI (Yapay Zeka Devrede)")
    print("="*50)

    stage_names = ["[DERIN UYKU]", "[HAFIF UYKU]", "[YARI UYANIK]", "[YATAKTAN KALKTI]"]

    while time_to_target <= 10:
        norm_time = max(-1.0, min(1.0, time_to_target / 30.0))
        state = [norm_time, float(sleep_stage), sleep_inertia, light_level, sound_level, float(snooze_count), total_sleep_hours]
        action = predict(state)

        # Durum guncelleme
        if action == 1:
            light_level = min(1.0, light_level + 0.35)
            if sleep_stage == 0 and light_level >= 0.5:
                sleep_stage = 1
                sleep_inertia = max(0.1, sleep_inertia - 0.25)
        elif action == 2:
            sound_level = 0.35
            if sleep_stage == 1:
                sleep_stage = 2
                sleep_inertia = max(0.1, sleep_inertia - 0.2)
        elif action in [3, 4]:
            sound_level = 0.8
            sleep_stage = 3

        time_str = f"{time_to_target:+d} dk"
        print(f"\\n[Saat: {time_str:>7}] | Evre: {stage_names[sleep_stage]}")
        print(f"   * Ekran Isigi: %{int(light_level*100):<3} | Ses: %{int(sound_level*100):<3} | Sersemlik: %{int(sleep_inertia*100)}")
        print(f"   * Yapay Zeka Karari: {ACTION_NAMES[action]}")

        # Telefon hoparlorunden bip sesi
        if sound_level > 0:
            print("\\a")

        if sleep_stage == 3:
            print("\\n" + "="*50)
            print(f"[OK] BASARILI! Kullanici {time_to_target:+d}. dakikada YATAKTAN KALKTI!")
            print(f"   Kalan Uyku Sersemligi: %{int(sleep_inertia*100)} (Dinc Uyanis)")
            print("="*50)
            break

        time_to_target += 5
        time.sleep(step_delay)

def main():
    print("="*50)
    print("   AKILLI UYANMA ASISTANI (PYDROID 3 MOBIL SURUM)")
    print("="*50)
    print("1 - Hemen Canli Test Et (3 saniyelik adimlarla calistir)")
    print("2 - Yarin Sabaha Alarm Kur (Hedef saate 30 dk kala baslar)")
    
    secim = input("\\nSeciminiz (1 veya 2): ").strip()
    if secim == "1":
        run_simulation(fast_mode=True)
    elif secim == "2":
        hedef = input("Hedef Uyanma Saati (Orn: 07:30): ").strip()
        try:
            h, m = map(int, hedef.split(":"))
            now = datetime.datetime.now()
            target_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if target_dt <= now:
                target_dt += datetime.timedelta(days=1)
            
            # Hedefe 30 dk kala basla
            start_dt = target_dt - datetime.timedelta(minutes=30)
            wait_seconds = (start_dt - now).total_seconds()

            print(f"\\n[OK] Alarm Kuruldu! Hedef Saat: {hedef}")
            print(f"Yapay zeka saat {start_dt.strftime('%H:%M')}'de (30 dk kala) devreye girecek.")
            print("Telefonun ekranini kapatmadan sarjda birakabilirsiniz.")
            
            if wait_seconds > 0:
                print(f"Bekleniyor... ({int(wait_seconds // 3600)} saat {int((wait_seconds % 3600) // 60)} dakika)")
                time.sleep(wait_seconds)
            
            run_simulation(fast_mode=False)
        except Exception as e:
            print("Gecersiz saat formati! Lutfen 07:30 gibi girin.", e)
    else:
        print("Gecersiz secim!")

if __name__ == "__main__":
    main()
'''

out_path = os.path.join(BASE_DIR, "mobile_alarm.py")
with open(out_path, "w", encoding="utf-8") as out:
    out.write(header + body)

print(f"mobile_alarm.py successfully created at {out_path}!")
