# 🌅 Pekiştirmeli Öğrenme ile Akıllı Uyanma Asistanı (Smart Wake-Up RL Agent)

Geleneksel telefon alarmlarının körü körüne çalarak neden olduğu **uyku sersemliğini (sleep inertia)** ve kronikleşen **erteleme (snooze)** krizini, **Pekiştirmeli Öğrenme (Reinforcement Learning - Deep Q-Network)** kullanarak çözen kişisel yapay zeka projesi.

---

## 🎯 Problem ve Biyolojik Gerçeklik
Standart alarmlar saat 07:00'de birden son ses çalar. Eğer kullanıcı o sırada **derin uyku (deep sleep)** evresindeyse beyin şoka girer (*sleep inertia*), refleks olarak "Ertele" tuşuna basar ve döngü başlar.

**Yapay Zeka Çözümü:**
Ajan, hedef uyanma saatinden önceki 30 dakikalık pencerede uyku evresini takip eder. Önce **ışığı kademeli artırarak melatonin hormonunu baskılar**, derin uykudan hafif uykuya geçiş sağlandığında **kısık melodi ve standart uyarılarla** kişiyi sıfır erteleme ile dinç şekilde yataktan kaldırır.

---

## 🧭 Markov Karar Süreci (MDP) Modellemesi

| Bileşen | Detay |
| :--- | :--- |
| **Durum Uzayı ($S$)** | 7 Boyutlu Vektör: Kalan Süre, Uyku Evresi, Uyku Sersemliği, Işık Seviyesi, Ses Seviyesi, Erteleme Sayısı, Toplam Uyku Süresi |
| **Eylem Uzayı ($A$)** | 5 Ayrık Eylem: `0: Bekle`, `1: Işığı Artır`, `2: Hafif Melodi`, `3: Standart Alarm`, `4: Agresif Mod` |
| **Ödül Fonksiyonu ($R$)** | Hedef saatte kalkış: `+100`, Dinçlik bonusu: `+(1-Inertia)*35`, Erteleme cezası: `-15/kez`, Geç kalma cezası: `-100` |
| **Algoritma** | **Deep Q-Network (DQN)** + Experience Replay (15.000 bellek) + Target Network + Epsilon-Greedy |

---

## 📊 Kıyaslama Sonuçları (100 Sabah Simülasyonu - Yeni Anti-Hacking Model)

| Metrik | Geleneksel Sabit Alarm | RL Akıllı Uyanma Asistanı (DQN) | Fark / Kazanç |
| :--- | :---: | :---: | :---: |
| **Zamanında Uyanma Başarısı** | %32.0 | **%100.0** | **+%68 Daha Yüksek** |
| **Ortalama Erteleme (Snooze)** | 1.54 kez | **0.00 kez** | **Sıfır Erteleme!** |
| **Kalan Uyku Sersemliği (Inertia)** | %81.5 (Ağır Sersemlik) | **%10.9 (Tam Dinç)** | **%86 Daha Dinç Uyanma** |
| **Ortalama Toplam Ödül** | -74.4 | **+138.3** | **Optimal Politika** |

---

## 🚀 Projeyi Çalıştırma

### 1. Modeli Eğitme ve Grafikleri Üretme:
```bash
python train.py
```
*(Eğitilen model `models/best_wake_up_agent.pth` dosyasına, grafikler `plots/training_results.png` dosyasına kaydedilir.)*

### 2. Kıyaslama Testini Çalıştırma:
```bash
python evaluate.py
```

### 3. Otomatik Test Paketini Çalıştırma (17 Adet Birim & Stres Testi):
```bash
pytest test_wake_up_project.py -v
```

### 4. İnteraktif Web Arayüzünü Başlatma (Streamlit Dashboard & Q-Explorer):
```bash
streamlit run app.py
```

### 5. Sıfır Bağımlılık Mobil / Pydroid 3 Sürümü:
```bash
python generate_mobile.py
python mobile_alarm.py
```

---

## 📂 Dosya Mimarisi
```text
6-Proje-Akilli-Uyanma-Asistani-RL/
│
├── README.md                    # Proje tanıtım ve dokümantasyon
├── PROJE_PLANI_VE_TODO.md       # Adım adım faz takip listesi
├── wake_up_env.py               # Gymnasium özel simülasyon ortamı (Biyolojik uyku dinamikleri)
├── dqn_agent.py                 # PyTorch DQN, Replay Buffer & Target Network mimarisi
├── train.py                     # 600 sabah simülasyonunda model eğitimi ve ağırlık dışa aktarımı
├── evaluate.py                  # Klasik alarm vs. DQN kıyaslama scripti
├── app.py                       # Streamlit görsel web arayüzü ve Q-değeri açıklanabilirlik masası
├── test_wake_up_project.py      # 17 adet kapsamlı birim, stres, mobil ve ağırlık tutarlılık testi
├── generate_mobile.py           # Saf Python / Pydroid 3 tek dosya mobil betiği üreticisi
├── mobile_alarm.py              # Harici kütüphane gerektirmeyen bağımsız mobil uyanma scripti
├── models/                      # Eğitilmiş PyTorch (.pth), TorchScript (.pt) ve JSON ağırlıkları
├── plots/                       # Eğitim analiz ve performans grafikleri
└── android/                     # Tam teşekküllü Native Kotlin Android Studio projesi
    ├── app/src/main/assets/     # Model ağırlıkları (model_weights.json)
    ├── app/src/main/java/       # Kotlin Yapay Zeka (DQN), Servis ve UI kodları
    └── README_ANDROID.md        # Android çalıştırma ve kurulum rehberi
```
