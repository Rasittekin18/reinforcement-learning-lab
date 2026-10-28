# Reinforcement Learning: Mathematical Foundations, Deep Q-Networks & Custom Environment Design Handbook

Bu doküman, Pekiştirmeli Öğrenme (Reinforcement Learning) disiplinindeki matematiksel temelleri (Markov Karar Süreçleri, Bellman Eşitlikleri), Tabular Q-Learning algoritmalarını, Deep Q-Networks (DQN) mimarisini, özel Pygame tabanlı simülasyon/ortam tasarımını ve evrişimli (CNN) Deep Convolutional Q-Learning (DCQL) yöntemlerini detaylı şekilde açıklayan teknik bir referans ve uygulama el kitabıdır.

---

## 🧭 Mimari ve Gelişim Yol Haritası

```
+---------------------------------------------------------------------------------------------------+
| 1. RL'e Giriş & Temel Kavramlar (Ajan, Ortam, Durum, Eylem, Ödül, Markov Karar Süreci)             |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 2. Tabular Q-Learning (Bellman Denklemi, Epsilon-Greedy, TD-Learning | Projeler: Taxi-v2, FrozenLake)|
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 3. Deep Q-Learning - DQL (Yapay Sinir Ağları, Experience Replay | Projeler: CartPole, LunarLander)|
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 4. Özel Ortam Tasarımı (Pygame, Sprite Sistemi, Çarpışma, Env Sınıfı | Proje: DQL Kaçış Oyunu)     |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 5. Deep Convolutional Q-Learning - DCQL (CNN, Frame Stacking, Piksel Girdisi | Proje: Atari Pong) |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 6. Bitirme Projesi: Akıllı Uyanma Asistanı (Sirkadiyen Biyoloji, Anti-Hacking, Streamlit & Android)|
+---------------------------------------------------------------------------------------------------+
```

---

## BÖLÜM 1: Reinforcement Learning (Pekiştirmeli Öğrenme) Temelleri

### 1.1. Makine Öğrenmesinde RL'in Yeri
* **Supervised Learning (Denetimli Öğrenme):** Etiketli veri vardır ($X \rightarrow y$). Model doğru cevabı görerek öğrenir.
* **Unsupervised Learning (Denetimsiz Öğrenme):** Etiketsiz verideki örüntüler, gruplar ve kümeler bulunur.
* **Reinforcement Learning (Pekiştirmeli Öğrenme):** Veri veya etiket yoktur. Bir **Ajan (Agent)**, bir **Ortam (Environment)** içinde deneme-yanılma (trial & error) yaparak, aldığı **Ödül (Reward)** ve **Cezalar (Penalty)** doğrultusunda kümülatif getirisini maksimize eden politikayı (davranış biçimini) öğrenir.

### 1.2. Temel Kavramlar Sözlüğü
1. **Agent (Ajan):** Karar verici ve eylemleri gerçekleştiren varlık (örn. Taksi şoförü, oyun karakteri, robot kolu).
2. **Environment (Ortam):** Ajanın içinde bulunduğu, durumlara ve eylemlere tepki veren dünya (örn. Izgara harita, oyun ekranı).
3. **State ($S$ / Durum):** Ajanın o anda bulunduğu koşul veya ortamın anlık temsili.
4. **Action ($A$ / Eylem):** Ajanın bir durumda yapabileceği hamleler (örn. İleri git, zıpla, yolcu al).
5. **Reward ($R$ / Ödül):** Ajanın yaptığı eylem sonucunda ortamdan aldığı skalar geri bildirim (pozitif veya negatif).
6. **Policy ($\pi$ / Politika):** Ajanın hangi durumda hangi eylemi seçeceğini belirleyen strateji haritası ($\pi(s) \rightarrow a$).
7. **Episode (Bölüm):** Oyunun başlangıcından bitiş durumuna (terminal state) kadar geçen tüm adımlar.
8. **Living Penalty (Yaşam Cezası):** Ajanın hedefe boş yere oyalanmadan en kısa yoldan gitmesini sağlamak için her adımda verilen küçük negatif ödül (örn: her adımda -1).

### 1.3. Gerçek Hayat ve Başarı Örnekleri
* **AlphaGo (DeepMind):** Dünya Go şampiyonunu yenen sistem.
* **OpenAI Five:** Dota 2 oyununda profesyonel takımları yenen DQL tabanlı ajanlar.
* **Roket İnişi (Landing a Rocket / SpaceX):** Roketlerin dikey iniş kontrol algoritmaları.
* **Endüstriyel Robotik:** Otomatik montaj ve fabrika optimizasyonları.

---

## BÖLÜM 2: Tabular Q-Learning (Tablo Tabanlı Q-Öğrenme)

### 2.1. Markov Karar Süreci (MDP) ve Ortam Türleri
* **MDP (Markov Decision Process):** Bir durumdaki gelecekteki geçişler, yalnızca o anki duruma ($s$) ve yapılan eyleme ($a$) bağlıdır; geçmişin geçmişine bağımlı değildir (Hafızasızlık özelliği).
* **Deterministic Ortam:** Ajan "Yukarı" dediğinde kesinlikle %100 yukarı gider.
* **Stochastic (Olasılıksal) Ortam:** Ajan "Yukarı" gitmek istese bile rüzgar veya zemin kayması nedeniyle %80 yukarı, %10 sağa, %10 sola kayabilir (Örn: `FrozenLake`).

### 2.2. Bellman Denklemi ve Temporal Difference (TD)
Q-değeri ($Q(s, a)$), $s$ durumunda $a$ eylemini yapmanın gelecekteki beklenen toplam indirimli getirisidir:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

* **$\alpha$ (Learning Rate - Öğrenme Oranı):** Yeni bilginin eskisinin üzerine ne kadar yazılacağı ($0 < \alpha \le 1$).
* **$\gamma$ (Discount Factor - İndirim Faktörü):** Gelecekteki ödüllerin bugünkü değerini belirler ($0 \le \gamma < 1$). $\gamma \to 1$ ise ajan vizyonerdir, gelecekteki büyük ödülü önemser. $\gamma \to 0$ ise ajan miyoptur, anlık ödüle bakar.
* **TD Error (Temporal Difference Hatası):** $\left[ R + \gamma \max_{a'} Q(s', a') \right] - Q(s, a)$ ifadesidir. Gerçekleşen getiri ile beklenen değer arasındaki farktır.

### 2.3. Keşif ve Sömürü (Exploration vs. Exploitation)
* **Exploration (Keşif):** Bilinmeyen durumları keşfetmek için rastgele eylem seçmek.
* **Exploitation (Sömürü):** Mevcut tecrübeye göre bilinen en yüksek Q-değerine sahip eylemi seçmek (`argmax`).
* **$\epsilon$-Greedy Yaklaşımı:**
  $$\text{Action} = \begin{cases} \text{Rastgele Eylem (Sample)}, & \text{eğer } \text{rand}() < \epsilon \\ \arg\max_a Q(s, a), & \text{aksi takdirde} \end{cases}$$

### 2.4. Projeler ve Uygulama Kodları

#### 1) OpenAI Gym Taxi-v2 / Taxi-v3
* **Durum Uzayı (State Space):** 500 durum ($5 \times 5 \text{ ızgara} \times 5 \text{ yolcu konumu} \times 4 \text{ hedef} = 500$).
* **Eylemler (Action Space - 6 Adet):** 0: Güney, 1: Kuzey, 2: Doğu, 3: Batı, 4: Yolcu Al (Pickup), 5: Yolcu Bırak (Dropoff).
* **Ödüller:** Başarılı bırakma: `+20`, Yanlış bırakma/alma: `-10`, Her geçen adım (Living penalty): `-1`.
* **Kritik Kod Yapısı:**
  ```python
  q_table = np.zeros([env.observation_space.n, env.action_space.n])
  for episode in range(10000):
      state = env.reset()
      while True:
          if random.uniform(0, 1) < epsilon:
              action = env.action_space.sample() # Explore
          else:
              action = np.argmax(q_table[state]) # Exploit
          
          next_state, reward, done, _ = env.step(action)
          
          # Q-Learning formülü
          old_val = q_table[state, action]
          next_max = np.max(q_table[next_state])
          q_table[state, action] = (1 - alpha) * old_val + alpha * (reward + gamma * next_max)
          
          state = next_state
          if done: break
  ```

#### 2) FrozenLake-v0 (Buz Gölü)
* **Harita:** 4x4 matris. S: Başlangıç, F: Donmuş zemin, H: Delik (Ölüm, reward=0), G: Hedef (Başarı, reward=1).
* **Zorluk:** `is_slippery=True` iken ortam stokastiktir. Ajan hedefe giderken kayıp deliğe düşebilir. Çözüm için daha yüksek bölüm sayısı (örn: 75.000) ve optimize edilmiş $\gamma, \alpha$ parametreleri kullanılır.

---

## BÖLÜM 3: Deep Q-Learning (DQL - Derin Q-Öğrenme)

### 3.1. Neden Deep Q-Learning? (Boyut Laneti - Curse of Dimensionality)
Tabular Q-learning'de her durum bir satırdır. Ancak durumlar sürekli (continuous) olduğunda (örn: açı $0.1523^\circ$, hız $1.432$ m/s) veya durum uzayı trilyonları bulduğunda tablo tutmak imkansızdır.
**Çözüm:** Q-tablosu yerine bir **Yapay Sinir Ağı (ANN)** kullanarak durum vektöründen ($S$) tüm eylemlerin tahmini Q değerlerini ($Q(s, a)$) çıktı olarak üreten bir fonksiyon yaklaşıklayıcı (Function Approximator) inşa etmektir.

### 3.2. DQL'in İki Temel Kararsızlık Sorunu ve Çözümleri
1. **Experience Replay (Deneyim Tekrarı):**
   * *Problem:* Ajan ardışık adımlarda öğrendikçe veriler zaman içinde birbiriyle yüksek derecede ilişkili (correlated) olur. Bu da ağın ağırlıklarını saptırır.
   * *Çözüm:* Geçişler `(state, action, reward, next_state, done)` bir hafızada (`collections.deque`) saklanır. Eğitim sırasında bu hafızadan rastgele küçük gruplar (`minibatch`, örn: 16 veya 32) çekilerek veri bağımlılığı kırılır.
2. **Target Network (Hedef Ağ Çözümü - Moving Target Problem):**
   * *Problem:* Ağ kendi tahmin ettiği değerleri yine kendi güncel ağırlıklarıyla hesaplanan hedeflere uydurmaya çalışırsa hedef sürekli hareket eder ve model ıraksar.
   * *Çözüm:* Biri anlık eğitilen (`model`), diğeri ağırlıkları periyodik olarak kopyalanan (`target_model`) iki ayrı ağ kullanılır.
     $$\text{Target} = R + \gamma \max_{a'} Q_{\text{target}}(S', a')$$

### 3.3. Adaptive Epsilon-Greedy
Ajan başlangıçta çevreyi tanımak için çok keşif yapmalı ($\epsilon = 1.0$), öğrendikçe sömürüye geçmelidir:
$$\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \times \epsilon_{decay}) \quad (\text{Örn: } \epsilon_{decay} = 0.995, \epsilon_{min} = 0.01)$$

### 3.4. DQL Projeleri

#### 1) CartPole-v0 (Ters Sarkaç Dengeleme)
* **Amaç:** Hareketli bir araba üzerindeki dik çubuğu devirmeden dengede tutmak.
* **Gözlem (4 Değer):** Araba konumu, araba hızı, çubuk açısı, çubuk açısal hızı.
* **Eylemler (2 Adet):** 0: Sola it, 1: Sağa it.
* **Ağ Mimarisi:**
  * Giriş: `Dense(48, input_dim=4, activation='tanh')`
  * Çıkış: `Dense(2, activation='linear')`
  * Kayıp / Optimizasyon: MSE + Adam optimizer (`lr=0.001`).

#### 2) LunarLander-v2 (Ay Modülü İnişi)
* **Gözlem (8 Boyutlu Vektör):** Koordinatlar, hızlar, açı, bacakların yere temas durumu.
* **Eylemler (4 Adet):** Hiçbir şey yapma, sol motor, ana motor, sağ motor.
* **Önemli İyileştirme:** Vektörize edilmiş (hızlı numpy) replay metodu ve `target_model` kullanımı.

---

## BÖLÜM 4: Reinforcement Learning Ortam Tasarımı (Custom Environment Design)

Her problem için hazır OpenAI Gym ortamı bulunmaz. Bu modülde **Pygame** kullanılarak sıfırdan bir RL simülasyon ve etkileşim ortamının tasarlanması ele alınmaktadır.

### 4.1. Pygame Oyun Geliştirme Mimarisi
Bir oyunun temel omurgası **Game Loop** (Oyun Döngüsü) mimarisidir:
1. **Process Input (Girdi Toplama):** Klavye/fare veya Ajanın eylemleri.
2. **Update Game (Durum Güncelleme):** Varlıkların konumlarını, hızlarını ve çarpışmalarını güncelleme.
3. **Render / Draw (Ekrana Çizme):** Kareyi çizme ve ekrana yansıtma (`pygame.display.flip()`).
4. **Clock Tick:** FPS kontrolü (`clock.tick(30)`).

### 4.2. OOP ve Sprite Sistemi
* `pygame.sprite.Sprite`: Oyuncu, düşman ve nesnelerin türetildiği ana sınıf.
* `pygame.sprite.Group`: Tüm nesneleri topluca güncelleyen (`update()`) ve çizen (`draw()`) yapı.
* `pygame.sprite.spritecollide`: Nesneler arasındaki daire veya kutu çarpışmalarını algılar.

### 4.3. Özel RL Kaçış Oyunu (Evade Falling Enemies)
* **Oyun Mantığı:** Aşağıda duran bir Ajan (`Player`), yukarıdan rastgele koordinatlardan düşen 2 adet düşmandan (`Enemy`) kaçmaya çalışır.
* **State Vektörü (4 Eleman):**
  $$S = [\Delta X_1, \Delta Y_1, \Delta X_2, \Delta Y_2]$$
  Ajanın 1. ve 2. düşmana olan yatay ve dikey uzaklıkları ($X_{player} - X_{enemy}$, $Y_{player} - Y_{enemy}$).
* **Eylem Uzayı (3 Eylem):** 0: Sola git (-4 px), 1: Sağa git (+4 px), 2: Dur (0 px).
* **Ödül Tasarımı:**
  * Hayatta kalınan her an: `+2` ödül.
  * Düşmana çarpıldığında: `-150` ceza ve bölüm sonu (`done = True`).
* **RL Entegrasyonu:** `Env` sınıfı `step(action)` ve `initialStates()` metotlarıyla OpenAI Gym benzeri bir arayüz sunar ve DQL Agent bu ortamı kendi kendine oynayarak düşmanlardan kaçmayı öğrenir.

---

## BÖLÜM 5: Deep Convolutional Q-Learning (DCQL - Görsel Girdili Derin RL)

### 5.1. Neden CNN (Evrişimli Sinir Ağları)?
Durum uzayı sayısal koordinatlar yerine **ham piksel matrislerinden (ekran görüntüsü)** oluştuğunda normal yoğun (Dense) ağlar piksel ilişkilerini ve mekansal (spatial) özellikleri kaybeder. CNN katmanları pikseller arasındaki kenar, şekil ve konum ilişkilerini öğrenir.

### 5.2. Ham Görüntü Ön İşleme (Preprocessing Hattı)
Oyun ekranı çok büyük ve gereksiz renk detaylarıyla doludur. `skimage` ile şu adımlar uygulanır:
1. **Griye Dönüştürme (`rgb2gray`):** Renk bilgisi gereksizdir; boyut 3 kanaldan 1 kanala iner.
2. **Kırpma (Cropping):** Skor tablosu gibi gereksiz bölgeler kırpılır (örn: 400x400).
3. **Yeniden Boyutlandırma (Resize):** Modelin hızlı eğitilmesi için görüntü 40x40 piksele küçültülür.
4. **Normalizasyon:** Görüntü matrisi piksel yoğunluklarına göre normalize edilir ($/ 128$).

### 5.3. Frame Stacking (Kare İstifleme) ve Zamansal Dinamikler
* **Kritik Kavram:** Tek bir durağan kareye bakarak tenis topunun **nereye doğru gittiğini** veya **hızını** anlayamazsınız.
* **Çözüm:** Son 4 kare (`IMGHISTORY = 4`) kanal boyutunda üst üste istiflenir:
  $$\text{Girdi Boyutu} = (40, 40, 4)$$
  Böylece evrişim katmanları topun yönünü ve ivmesini pikseller arasındaki değişimden yakalar.

### 5.4. DCQL Mimarisi ve Pong Projesi
* **Ortam:** Klasik Pong (Tenis) oyunu. Sol raket Ajan, sağ raket kural tabanlı yapay zeka.
* **Eylemler (3 Adet):** 0: Sabit dur, 1: Yukarı hareket et, 2: Aşağı hareket et.
* **Ödüller:** Topu karşılayıp vurunca `+10`, top kaçırılınca `-10`, her geçen kare `-0.05`.
* **CNN Mimarisi:**
  1. `Conv2D(32, kernel=(4,4), strides=(2,2), padding="same", activation="relu")`
  2. `Conv2D(64, kernel=(4,4), strides=(2,2), padding="same", activation="relu")`
  3. `Conv2D(64, kernel=(3,3), strides=(1,1), padding="same", activation="relu")`
  4. `Flatten()` $\to$ `Dense(512, activation="relu")`
  5. `Dense(3, activation="linear")`
* **Gözlem Süresi (Observe Period):** İlk 2.000 adımda model ağırlıkları eğitilmez (`OBSERVEPERIOD = 2000`), sadece rastgele hamlelerle replay hafızası doldurulur. Ardından $\epsilon$ kademeli olarak $1.0 \to 0.05$ seviyesine kadar düşürülür.

---

## BÖLÜM 6: Bitirme Projesi - Pekiştirmeli Öğrenme ile Akıllı Uyanma Asistanı (Capstone)

### 6.1. Problem Tanımı ve Biyolojik Gerçeklik
Geleneksel saat alarmları kullanıcının uyku evresini (Derin Uyku, Hafif Uyku, REM) gözetmeksizin doğrudan ses çıkarır. Kullanıcı derin uykudayken uyarılırsa şiddetli **uyku sersemliği (sleep inertia)** yaşar ve kronik **erteleme (snooze)** döngüsüne girer.

### 6.2. MDP ve Anti-Hacking Ödül Modellemesi
* **Durum Uzayı ($S \in \mathbb{R}^7$):**
  1. `norm_time_to_target`: Hedefe kalan süre (-30 dk ile +15 dk arası normalize edilmiş).
  2. `sleep_stage`: Anlık uyku evresi (0: Derin, 1: Hafif, 2: Yarı Uyanık, 3: Kalktı).
  3. `sleep_inertia`: Kalan uyku sersemliği katsayısı ($0.0 \sim 1.0$).
  4. `light_level`: Ortam/ekran ışığı seviyesi ($0.0 \sim 1.0$).
  5. `sound_level`: Ses düzeyi ($0.0 \sim 1.0$).
  6. `snooze_count`: Kümülatif erteleme sayısı ($0 \sim 5$).
  7. `total_sleep_hours`: Gece uyunan toplam süre (Domain randomization: $4.5 \sim 8.5$ saat).
* **Eylem Uzayı ($A \in \{0, 1, 2, 3, 4\}$):**
  * `0: Bekle (Do Nothing)` - Uyku döngüsünü bölme.
  * `1: Işığı Kademeli Artır` - Melatonin hormonunu baskılayıp derin uykudan hafife geçiş sağla.
  * `2: Hafif Melodi / Doğa Sesi` - Yarı uyanıklığa yumuşak geçiş.
  * `3: Standart Alarm` - Ritmik ses ile uyandır.
  * `4: Agresif Mod` - Maksimum ses ve ışık (Yalnızca acil son çare).
* **Anti-Hacking Ödül Fonksiyonu ($R$):**
  * Hedef saat penceresinde (-5 ile +5 dk) kalkış: `+120`
  * Erken agresif mod açma cezası ($t < 0$): `-45` (Ajan insanı gürültüyle terörize edemez)
  * Erken standart alarm cezası ($t < -15$): `-25`
  * Erteleme (snooze) cezası: `-15` / kez
  * Dinç kalkma bonusu: `+(1 - Inertia) * 40`
  * Sıfır erteleme bonusu: `+30`

### 6.3. Çok Platformlu Dağıtım ve Entegrasyon
1. **PyTorch DQN Ajanı:** Target network + Huber Loss + Experience Replay (20.000 bellek).
2. **Streamlit İnteraktif Dashboard:** Canlı simülasyon, Q-değeri açıklanabilirlik masası ve 100 sabah kıyaslama testi.
3. **Pydroid 3 Tek Dosya Mobil:** Saf Python matris çıkarımı ile harici kütüphane gerektirmeyen bağımsız çalıştırma.
4. **Native Android (Kotlin):** 0.05 ms çıkarım süreli saf Kotlin matris motoru, AlarmManager, Foreground Service ve gün doğumu kilit ekranı arayüzü.

---

## 📊 Karşılaştırmalı Özet Tablosu

| Yöntem | Durum Temsili ($S$) | Model Türü | Bellek / Hafıza | Kullanım Alanı | Proje / Benchmark |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q-Learning** | Ayrık Sayılar / İndeksler (Örn: 0-499) | 2 Boyutlu Dizi ($|S| \times |A|$) | Gerek yok | Küçük, ayrık durum uzayları | Taxi-v2, FrozenLake |
| **Deep Q-Learning (DQL)** | Sürekli Vektörler (Örn: Açı, Hız, Mesafe) | Çok Katmanlı İleri Beslemeli ANN | Experience Replay (`deque`) | Sürekli/büyük durum uzayları | CartPole, LunarLander, Kaçış Oyunu |
| **Deep Conv Q-Learning (DCQL)** | Piksel Kareleri Matrisi (Örn: $40 \times 40 \times 4$) | Evrişimli Sinir Ağı (CNN) | Experience Replay (`deque`) | Görsel ekran girdisiyle oynanan oyunlar | Atari Pong |
| **Smart Wake-Up (Capstone DQN)** | Biyolojik & Zamansal 7 Boyutlu Vektör | PyTorch DQN + Saf Mobil Matris | Experience Replay (20k) | Gerçek Dünya Biyomedikal Karar Destek | Akıllı Uyanma Asistanı (Android & Streamlit) |

---

## ⚡ Hızlı Tekrar ve Mülakat Hazırlığı (Kritik Soru - Cevap)

**S1: Bellman Denklemi sezgisel olarak ne anlatır?**  
> *"Bugünkü bir eylemin değeri, o eylemden anında aldığın ödül ile gelecekte ulaşacağın en iyi durumun bugüne indirimli değerinin toplamıdır."*

**S2: $\gamma$ (Discount Factor) 0 olursa ve 1'e yakın olursa ne olur?**  
> $\gamma = 0$ ise ajan sadece o anki adımdaki ödülü önemser (kısa vadeli). $\gamma \to 1$ ise ajan gelecekteki ödülleri bugünkü kadar değerli kabul eder (uzun vadeli planlama yapar).

**S3: DQL'de neden Experience Replay kullanmak zorundayız?**  
> Ardışık zaman adımlarında toplanan veriler birbiriyle yüksek korelasyonludur (non-i.i.d.). Deneyim tekrarı hafızasından rastgele örneklem alarak veriler arasındaki zamansal bağımlılık kırılır ve gradyan inişi kararlı hale getirilir.

**S4: DCQL'de neden tek kare yerine 4 kare (Frame Stacking) beslenir?**  
> Tek bir statik görüntü nesnelerin hareket yönü, hızı ve ivmesi hakkında bilgi vermez (Markovian eksiklik). 4 ardışık kare üst üste eklendiğinde nesnenin yönü ve hızı ağ tarafından öğrenilebilir hale gelir.

**S5: Epsilon-Greedy stratejisinde neden sabit epsilon yerine "decaying" (azalan) epsilon tercih edilir?**  
> Eğitimin başında ajan ortam hakkında hiçbir şey bilmez; bu yüzden yüksek keşif ($\epsilon = 1.0$) gerekir. Eğitim ilerledikçe ajan optimal politikayı öğrenir, bu yüzden keşfi azaltıp bildiği en iyi hareketleri sömürmeye ($\epsilon \to 0.01$) geçmelidir.
