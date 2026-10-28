# 🌅 Proje: Pekiştirmeli Öğrenme ile Akıllı Uyanma Asistanı (Smart Wake-Up RL Agent)

Bu proje, geleneksel alarmların körü körüne ve zamansız çalarak neden olduğu **uyku sersemliğini (sleep inertia)** ve kronikleşen **erteleme (snooze)** döngüsünü, **Pekiştirmeli Öğrenme (Reinforcement Learning - Deep Q-Network)** kullanarak çözen kişisel bir yapay zeka asistanıdır.

---

## 🎯 Projenin Temel Amacı
Ajan, kullanıcının hedef uyanma saatinden önceki 30-40 dakikalık zaman penceresinde, kullanıcının uyku evresini (Hafif Uyku, Derin Uyku, REM) ve yorgunluğunu gözlemleyerek; **ışık, ses ve kademeli uyarı eylemlerini** en uygun zamanlama ile devreye sokar. Amaç: **Sıfır erteleme ile dinç uyanmayı sağlamak.**

---

## 🧭 Proje Fazları ve Yol Haritası (To-Do List)

```
[Faz 1: Modelleme] ──► [Faz 2: Gym Ortamı] ──► [Faz 3: DQN Ajanı] ──► [Faz 4: Eğitim & Metrikler] ──► [Faz 5: Kıyaslama & Arayüz] ──► [Faz 6: Mobil Dağıtım] ──► [Faz 7: Kapsamlı Testler]
```

### 📌 Faz 1: Problem Modellemesi ve Mimari Tasarım
- [x] Problem tanımı ve biyolojik temeller (Uyku evreleri, melatonin, uyku ataleti)
- [x] Durum Uzayı ($S$), Eylem Uzayı ($A$) ve Ödül Fonksiyonu ($R$) formülasyonu
- [x] Proje dosya yapısının ve çalışma dizininin hazırlanması

### 📌 Faz 2: Özel Simülasyon Ortamı (`wake_up_env.py`)
- [x] `gymnasium.Env` temelinde `SmartWakeUpEnv` sınıfının oluşturulması
- [x] İnsan Uyku Dinamiği Simülatörü (Derin uyku, hafif uyku geçişleri, yorgunluk etkisi)
- [x] Erteleme (Snooze) ve uyanma olasılık matematik modeli
- [x] `step()`, `reset()`, `render()` metotlarının yazımı
- [x] Ortamın rastgele eylemlerle doğrulanması (Sanity Check)

### 📌 Faz 3: Deep Q-Network (DQN) Ajanı (`dqn_agent.py`)
- [x] PyTorch Q-Network & Target Network mimarisinin kurulması
- [x] Experience Replay Buffer (Deneyim Hafızası) mekanizması
- [x] Epsilon-Greedy (Keşif ve Sömürü) stratejisi ve decay planı
- [x] Bellman Denklemi & Temporal Difference (TD) Loss optimizasyonu

### 📌 Faz 4: Model Eğitimi & Performans Grafikleri (`train.py`)
- [x] Eğitim döngüsünün kodlanması (500 sabah simülasyonu)
- [x] Başarı metriklerinin takibi (Erteleme sayısı, hedefi tutturma oranı, toplam ödül)
- [x] Matplotlib ile eğitim grafikleri (Loss, Ödül ve Epsilon eğrileri)
- [x] En iyi model ağırlıklarının (`best_wake_up_agent.pth`) kaydedilmesi

### 📌 Faz 5: Kıyaslama, Simülasyon & İnteraktif Arayüz (`evaluate.py` & `app.py`)
- [x] **Klasik Alarm vs. RL Ajanı** karşılaştırmalı performans testi
- [x] Tek bir sabahın adım adım zaman çizelgesi simülasyonu (Zaman, Işık, Ses, Uyku Durumu)
- [x] Streamlit ile interaktif görsel demo (Kullanıcının kendi uyku saatini girip simüle edebileceği arayüz)
- [x] İnteraktif Q-Değeri Deney Masası ve Canlı 100 Sabah Kıyaslama Testi

### 📌 Faz 6: Mobil Dağıtım & Native Android Uygulaması (`android/` & `mobile_alarm.py`)
- [x] JSON model ağırlıkları dışa aktarımı (`model_weights.json`)
- [x] Pydroid 3 tek dosya saf Python uyanma motoru (`generate_mobile.py` & `mobile_alarm.py`)
- [x] Native Kotlin Android Studio projesi (`WakeUpAgent.kt`, saf matris çıkarımı)
- [x] Android `AlarmManager`, `WakeUpService` Foreground servisi ve gün doğumu kilit ekranı UI
- [x] Android 13/14+ izin uyumlulukları (`SCHEDULE_EXACT_ALARM`, `POST_NOTIFICATIONS`, `USE_FULL_SCREEN_INTENT`)

### 📌 Faz 7: Kapsamlı Test Paketi ve Doğrulama (`test_wake_up_project.py`)
- [x] 17 adet otomatik PyTest birim ve stres testi
- [x] Anti-hacking ceza doğrulaması ve politika dürüstlük kontrolleri
- [x] Saf Python mobil çıkarım ile PyTorch tensör eşleşme doğrulaması
- [x] JSON, PyTorch (.pth) ve Android Assets ağırlık senkronizasyon testleri

---

## 📂 Planlanan Klasör Mimarisi

```text
6-Proje-Akilli-Uyanma-Asistani-RL/
│
├── PROJE_PLANI_VE_TODO.md       # Bu plan ve ilerleme takip belgesi
├── wake_up_env.py               # Gymnasium özel çevre sınıfı
├── dqn_agent.py                 # PyTorch DQN & Replay Buffer mimarisi
├── train.py                     # Eğitimi başlatan ve ağırlıkları kaydeden script
├── evaluate.py                  # Klasik alarm vs. RL ajanı kıyaslama testi
├── app.py                       # Streamlit görsel arayüzü ve interaktif demo
└── models/                      # Eğitilmiş model dosyaları (.pth)
```
