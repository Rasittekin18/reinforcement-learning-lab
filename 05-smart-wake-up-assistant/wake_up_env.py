"""
SmartWakeUpEnv - Akıllı Uyanma Asistanı Gymnasium Ortamı (Gelişmiş & Gerçekçi Sürüm)
Reward Hacking ve Overfitting'i önleyen, biyolojik uyku döngülerini ve kademeli uyanmayı ödüllendiren simülatör.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SmartWakeUpEnv(gym.Env):
    """
    Kullanıcının sabah hedef saat penceresinde (-5 dk ile +5 dk arası),
    erteleme yapmadan ve uyku sersemliği (sleep inertia) yaşamadan dinç kalkmasını sağlayan RL ortamı.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Eylemler (Actions)
    ACTION_DO_NOTHING = 0        # Bekle (Mevcut durumu koru, uykuyu bölme)
    ACTION_INCREASE_LIGHT = 1    # Işığı kademeli artır (Gün doğumu / Melatonin baskılama)
    ACTION_SOFT_SOUND = 2        # Hafif ses / Doğa melodisi (Kısık ses)
    ACTION_ALARM_SOUND = 3       # Standart alarm sesi (Ritmik ses)
    ACTION_AGGRESSIVE_ALERT = 4  # Agresif mod (Maksimum ses ve ışık - Yalnızca son çare!)

    # Uyku Evreleri (Sleep Stages)
    STAGE_DEEP = 0    # Derin Uyku
    STAGE_LIGHT = 1   # Hafif Uyku / REM
    STAGE_DROWSY = 2  # Yatakta Yarı Uyanık
    STAGE_AWAKE = 3   # Yataktan Kalktı (Terminal Durum)

    # Kullanıcı Profilleri (Domain Randomization)
    PROFILE_LIGHT_SLEEPER = 0   # Işığa duyarlı, kolay uyanan
    PROFILE_NORMAL_SLEEPER = 1  # Standart kullanıcı
    PROFILE_HEAVY_SLEEPER = 2   # Ağır uyuyan (Erken ses gelirse inatla ertelemeye basan)

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Eylem Uzayı: 5 ayrık eylem
        self.action_space = spaces.Discrete(5)

        # Durum Uzayı (State Space - 7 boyutlu sürekli vektör):
        # 0: time_to_target (-30 dk ile +15 dk arası normalize edilmiş [-1.0, +1.0])
        # 1: sleep_stage (0: Derin, 1: Hafif, 2: Yarı Uyanık, 3: Kalktı)
        # 2: sleep_inertia (0.0: Dinç <-> 1.0: Aşırı sersem/yorgun)
        # 3: light_level (0.0: Zifiri karanlık <-> 1.0: Tam aydınlık)
        # 4: sound_level (0.0: Sessiz <-> 1.0: Maksimum ses)
        # 5: snooze_count (0 ile 5 arası)
        # 6: total_sleep_hours (Önceki gece uyunan toplam süre, 4.0 - 9.0 saat)
        low = np.array([-1.0, 0, 0.0, 0.0, 0.0, 0.0, 4.0], dtype=np.float32)
        high = np.array([1.0, 3, 1.0, 1.0, 1.0, 5.0, 9.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Simülasyon Değişkenleri
        self.current_step = 0
        self.max_steps = 9  # Her adım 5 dakika (Toplam 45 dk: -30 dk'dan +15 dk'ya)
        self.time_to_target = -30
        self.sleep_stage = self.STAGE_DEEP
        self.sleep_inertia = 0.5
        self.light_level = 0.0
        self.sound_level = 0.0
        self.snooze_count = 0
        self.total_sleep_hours = 7.0
        self.user_profile = self.PROFILE_NORMAL_SLEEPER

    def _get_obs(self):
        norm_time = np.clip(self.time_to_target / 30.0, -1.0, 1.0)
        return np.array([
            norm_time,
            float(self.sleep_stage),
            self.sleep_inertia,
            self.light_level,
            self.sound_level,
            float(self.snooze_count),
            self.total_sleep_hours
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.time_to_target = -30
        self.light_level = 0.0
        self.sound_level = 0.0
        self.snooze_count = 0

        # Her sabah farklı bir insan tipi ve uyku süresi simüle edilir (Overfitting önleyici)
        self.user_profile = int(self.np_random.choice([0, 1, 2], p=[0.25, 0.50, 0.25]))
        self.total_sleep_hours = float(self.np_random.uniform(4.5, 8.5))
        self.sleep_inertia = float(np.clip(1.0 - (self.total_sleep_hours / 10.0), 0.2, 0.85))

        # Başlangıçta uykunun derin veya hafif olma durumu
        self.sleep_stage = self.STAGE_DEEP if self.np_random.random() < 0.70 else self.STAGE_LIGHT

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        self.time_to_target += 5  # Her adım 5 dakika ilerler
        reward = -0.5  # Küçük adım maliyeti
        terminated = False
        truncated = False

        prev_stage = self.sleep_stage

        # --- 1. EYLEMİN ORTAMA ETKİSİ ---
        if action == self.ACTION_DO_NOTHING:
            self.sound_level = max(0.0, self.sound_level - 0.25)

        elif action == self.ACTION_INCREASE_LIGHT:
            self.light_level = min(1.0, self.light_level + 0.35)
            self.sound_level = max(0.0, self.sound_level - 0.15)
            # Kademeli gün doğumu ışığı doğru zamanda verilirse ödüllendirilir
            if self.time_to_target < -5 and self.sleep_stage in [self.STAGE_DEEP, self.STAGE_LIGHT]:
                reward += 3.0  # Akıllı ön hazırlık teşviki

        elif action == self.ACTION_SOFT_SOUND:
            self.sound_level = 0.35
            if self.sleep_stage == self.STAGE_LIGHT and -15 <= self.time_to_target <= 0:
                reward += 4.0  # Hafif uykuda yumuşak geçiş teşviki

        elif action == self.ACTION_ALARM_SOUND:
            self.sound_level = 0.75

        elif action == self.ACTION_AGGRESSIVE_ALERT:
            self.light_level = 1.0
            self.sound_level = 1.0

        # --- 2. REWARD HACKING ÖNLEYİCİ CEZALAR (Anti-Cheating) ---
        # Ajan hedef saatten çok önce agresif mod açarak insanı terörize ederse:
        if action == self.ACTION_AGGRESSIVE_ALERT and self.time_to_target < 0:
            reward -= 45.0  # Ağır rahatsızlık ve gürültü cezası!
            self.sleep_inertia = min(1.0, self.sleep_inertia + 0.4)

        # Hedef saate 15 dakikadan fazla varken standart alarm çalarsa:
        if action == self.ACTION_ALARM_SOUND and self.time_to_target < -15:
            reward -= 25.0  # Erken alarm cezası

        # --- 3. BİYOLOJİK UYKU DİNAMİĞİ VE KULLANICI TEPKİLERİ ---
        # Işık etkisi: Yüksek ışık melatonini baskılar ve derin uykudan hafife geçirir
        light_threshold = 0.40 if self.user_profile == self.PROFILE_LIGHT_SLEEPER else 0.60
        if self.light_level >= light_threshold and self.sleep_stage == self.STAGE_DEEP:
            transition_prob = 0.85 if self.user_profile == self.PROFILE_LIGHT_SLEEPER else 0.65
            if self.np_random.random() < transition_prob:
                self.sleep_stage = self.STAGE_LIGHT
                self.sleep_inertia = max(0.1, self.sleep_inertia - 0.25)  # Işıkla doğal gevşeme
                reward += 10.0  # Doğal evre geçişi ödülü

        # Ses ve Uyarı Tepkileri:
        if self.sleep_stage == self.STAGE_DEEP:
            if action in [self.ACTION_ALARM_SOUND, self.ACTION_AGGRESSIVE_ALERT]:
                # Derin uykudayken ani ses gelirse: Ağır uyuyanlar %90, normal %75 ertelemeye basar!
                snooze_prob = 0.90 if self.user_profile == self.PROFILE_HEAVY_SLEEPER else 0.75
                if self.np_random.random() < snooze_prob:
                    self.snooze_count += 1
                    reward -= 20.0  # Erteleme cezası
                    self.sleep_inertia = min(1.0, self.sleep_inertia + 0.35)
                else:
                    # Uyandı ama başı çatlıyor (Sleep inertia tavanda)
                    self.sleep_stage = self.STAGE_AWAKE
                    self.sleep_inertia = 0.90
                    reward -= 30.0  # Kötü ve sersem uyanma cezası

        elif self.sleep_stage == self.STAGE_LIGHT:
            # Hafif uykudayken yumuşak melodi veya artan ışık yarı uyanıklığa geçirir
            if action == self.ACTION_SOFT_SOUND or self.light_level >= 0.7:
                self.sleep_stage = self.STAGE_DROWSY
                self.sleep_inertia = max(0.1, self.sleep_inertia - 0.20)
                reward += 15.0  # Mükemmel nazik geçiş ödülü
            elif action in [self.ACTION_ALARM_SOUND, self.ACTION_AGGRESSIVE_ALERT]:
                # Hafif uykuda sert ses gelirse:
                if self.np_random.random() < 0.40:
                    self.snooze_count += 1
                    reward -= 15.0
                else:
                    self.sleep_stage = self.STAGE_AWAKE
                    self.sleep_inertia = 0.45

        elif self.sleep_stage == self.STAGE_DROWSY:
            # Kullanıcı yatakta yarı uyanık:
            if action in [self.ACTION_ALARM_SOUND, self.ACTION_AGGRESSIVE_ALERT]:
                # Doğru zamanda kalkış!
                self.sleep_stage = self.STAGE_AWAKE
            elif action == self.ACTION_SOFT_SOUND and self.time_to_target >= 0:
                if self.np_random.random() < 0.60:
                    self.sleep_stage = self.STAGE_AWAKE
            elif action == self.ACTION_DO_NOTHING and self.light_level < 0.4:
                # Karanlıkta bırakılırsa geri uyur (Relapse)
                if self.np_random.random() < 0.55:
                    self.sleep_stage = self.STAGE_LIGHT
                    reward -= 20.0  # Geri uyutma cezası

        # --- 4. BİTİŞ KOŞULLARI VE NİHAİ ÖDÜL ---
        if self.sleep_stage == self.STAGE_AWAKE:
            terminated = True
            time_diff = abs(self.time_to_target)

            if time_diff <= 5:
                # [MÜKEMMEL ZAMANLAMA]: Hedef saatte uyanış (-5 ile +5 dk arası)
                reward += 120.0
            elif self.time_to_target < -5:
                # [ERKEN UYANDIRMA CEZASI]: Uykuyu çaldığı için eksi puan!
                minutes_early = abs(self.time_to_target)
                reward -= (minutes_early * 3.5)  # Örn: 20 dk erken kaldırırsa -70 puan!
            else:
                # [GEÇ KALMA]:
                reward -= (self.time_to_target * 6.0)

            # Dinç kalkma bonusu (Düşük sleep inertia)
            reward += (1.0 - self.sleep_inertia) * 40.0

            # Ertelemesiz başarı bonusu
            if self.snooze_count == 0:
                reward += 30.0
            else:
                reward -= (self.snooze_count * 15.0)

        # Süre doldu ve kullanıcı hala yataktan kalkamadıysa:
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            reward -= 150.0  # İşe/Okula geç kalma felaket cezası

        return self._get_obs(), reward, terminated, truncated, {
            "time_to_target": self.time_to_target,
            "sleep_stage": self.sleep_stage,
            "sleep_inertia": self.sleep_inertia,
            "snooze_count": self.snooze_count,
            "user_profile": self.user_profile
        }

    def render(self):
        stage_names = ["[DERIN UYKU]", "[HAFIF UYKU]", "[YARI UYANIK]", "[YATAKTAN KALKTI]"]
        curr_stage = stage_names[self.sleep_stage]
        time_str = f"{self.time_to_target:+d} dk"
        print(f"[Zaman: {time_str:>7}] | Evre: {curr_stage:<18} | Isik: %{int(self.light_level*100):<3} | Ses: %{int(self.sound_level*100):<3} | Snooze: {self.snooze_count} | Sersemlik: %{int(self.sleep_inertia*100)}")
