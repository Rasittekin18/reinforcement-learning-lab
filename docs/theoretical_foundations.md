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

#### 2) FrozenLake-v0 / FrozenLake-v1 (Buz Gölü)
* **Harita:** 4x4 matris. S: Başlangıç, F: Donmuş zemin, H: Delik (Ölüm, reward=0), G: Hedef (Başarı, reward=1).
* **Zorluk:** `is_slippery=True` iken ortam stokastiktir. Ajan hedefe giderken kayıp deliğe düşebilir. Çözüm için optimize edilmiş $\gamma, \alpha$ parametreleri ve yeterli bölüm sayısı kullanılır.

---

## BÖLÜM 3: Deep Q-Learning (DQL - Derin Q-Öğrenme)

### 3.1. Neden Deep Q-Learning? (Boyut Laneti - Curse of Dimensionality)
Tabular Q-learning'de her durum bir satırdır. Ancak durumlar sürekli (continuous) olduğunda veya durum uzayı çok büyüdüğünde tablo tutmak imkansızdır.
**Çözüm:** Q-tablosu yerine bir **Yapay Sinir Ağı (ANN)** kullanarak durum vektöründen ($S$) tüm eylemlerin tahmini Q değerlerini ($Q(s, a)$) çıktı olarak üreten bir fonksiyon yaklaşıklayıcı (Function Approximator) inşa etmektir.

### 3.2. DQL'in İki Temel Kararsızlık Sorunu ve Çözümleri
1. **Experience Replay (Deneyim Tekrarı):**
   * *Problem:* Ajan ardışık adımlarda öğrendikçe veriler zaman içinde birbiriyle yüksek derecede ilişkili (correlated) olur. Bu da ağın ağırlıklarını saptırır.
   * *Çözüm:* Geçişler `(state, action, reward, next_state, done)` bir hafızada (`collections.deque`) saklanır. Eğitim sırasında bu hafızadan rastgele küçük gruplar (`minibatch`) çekilerek veri bağımlılığı kırılır.
2. **Target Network (Hedef Ağ Çözümü - Moving Target Problem):**
   * *Problem:* Ağ kendi tahmin ettiği değerleri yine kendi güncel ağırlıklarıyla hesaplanan hedeflere uydurmaya çalışırsa hedef sürekli hareket eder ve model ıraksar.
   * *Çözüm:* Biri anlık eğitilen (`model`), diğeri ağırlıkları periyodik olarak kopyalanan (`target_model`) iki ayrı ağ kullanılır.
     $$\text{Target} = R + \gamma \max_{a'} Q_{\text{target}}(S', a')$$

### 3.3. Adaptive Epsilon-Greedy
Ajan başlangıçta çevreyi tanımak için çok keşif yapmalı ($\epsilon = 1.0$), öğrendikçe sömürüye geçmelidir:
$$\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \times \epsilon_{decay})$$

### 3.4. DQL Projeleri

#### 1) CartPole-v0 / v1 (Ters Sarkaç Dengeleme)
* **Amaç:** Hareketli bir araba üzerindeki dik çubuğu devirmeden dengede tutmak.
* **Gözlem (4 Değer):** Araba konumu, araba hızı, çubuk açısı, çubuk açısal hızı.
* **Eylemler (2 Adet):** 0: Sola it, 1: Sağa it.

#### 2) LunarLander-v2 (Ay Modülü İnişi)
* **Gözlem (8 Boyutlu Vektör):** Koordinatlar, hızlar, açı, bacakların yere temas durumu.
* **Eylemler (4 Adet):** Hiçbir şey yapma, sol motor, ana motor, sağ motor.
* **Önemli İyileştirme:** Vektörize edilmiş replay metodu ve `target_model` kullanımı.

---

## BÖLÜM 4: Reinforcement Learning Ortam Tasarımı (Custom Environment Design)

Her problem için hazır OpenAI Gym ortamı bulunmaz. Bu modülde **Pygame** kullanılarak sıfırdan bir RL simülasyon ve etkileşim ortamının tasarlanması ele alınmaktadır.

### 4.1. Pygame Oyun Geliştirme Mimarisi
Bir oyunun temel omurgası **Game Loop** (Oyun Döngüsü) mimarisidir:
1. **Process Input (Girdi Toplama):** Ajanın eylemleri veya klavye/fare girdileri.
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
* **Eylem Uzayı (3 Eylem):** 0: Sola git, 1: Sağa git, 2: Dur.
* **Ödül Tasarımı:** Hayatta kalınan her an: `+2` ödül, düşmana çarpıldığında: `-150` ceza.

---

## BÖLÜM 5: Deep Convolutional Q-Learning (DCQL - Görsel Girdili Derin RL)

### 5.1. Neden CNN (Evrişimli Sinir Ağları)?
Durum uzayı sayısal koordinatlar yerine **ham piksel matrislerinden** oluştuğunda normal yoğun (Dense) ağlar piksel ilişkilerini ve mekansal (spatial) özellikleri kaybeder. CNN katmanları pikseller arasındaki kenar, şekil ve konum ilişkilerini öğrenir.

### 5.2. Ham Görüntü Ön İşleme (Preprocessing Hattı)
`skimage` ile şu adımlar uygulanır:
1. **Griye Dönüştürme (`rgb2gray`):** Renk bilgisi elenir; boyut 3 kanaldan 1 kanala iner.
2. **Kırpma (Cropping):** Skor tablosu gibi gereksiz bölgeler kırpılır (400x400).
3. **Yeniden Boyutlandırma (Resize):** Modelin hızlı eğitilmesi için görüntü 40x40 piksele küçültülür.
4. **Normalizasyon:** Görüntü matrisi piksel yoğunluklarına göre normalize edilir ($/ 128$).

### 5.3. Frame Stacking (Kare İstifleme) ve Zamansal Dinamikler
Tek bir durağan kare nesnelerin hızını ve hareket yönünü vermez. Son 4 kare (`IMGHISTORY = 4`) kanal boyutunda üst üste istiflenir: $\text{Girdi Boyutu} = (40, 40, 4)$. Böylece evrişim katmanları topun yönünü ve ivmesini yakalar.

### 5.4. DCQL Mimarisi ve Pong Projesi
* **Ortam:** Klasik Pong oyunu. Sol raket Ajan, sağ raket kural tabanlı yapay zeka.
* **Eylemler (3 Adet):** 0: Sabit dur, 1: Yukarı hareket et, 2: Aşağı hareket et.
* **Ödüller:** Topu karşılayıp vurunca `+10`, top kaçırılınca `-10`, her geçen kare `-0.05`.
* **Gözlem Süresi (Observe Period):** İlk adımlarda model eğitilmez (`OBSERVEPERIOD = 2000`), rastgele hamlelerle replay hafızası doldurulur. Ardından $\epsilon$ kademeli olarak düşürülür.

---

## 📊 Karşılaştırmalı Özet Tablosu

| Yöntem | Durum Temsili ($S$) | Model Türü | Bellek / Hafıza | Kullanım Alanı | Proje / Benchmark |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q-Learning** | Ayrık Sayılar / İndeksler (Örn: 0-499) | 2 Boyutlu Dizi ($|S| \times |A|$) | Gerek yok | Küçük, ayrık durum uzayları | Taxi-v2, FrozenLake |
| **Deep Q-Learning (DQL)** | Sürekli Vektörler (Örn: Açı, Hız, Mesafe) | Çok Katmanlı İleri Beslemeli ANN | Experience Replay (`deque`) | Sürekli/büyük durum uzayları | CartPole, LunarLander, Kaçış Oyunu |
| **Deep Conv Q-Learning (DCQL)** | Piksel Kareleri Matrisi (Örn: $40 \times 40 \times 4$) | Evrişimli Sinir Ağı (CNN) | Experience Replay (`deque`) | Görsel ekran girdisiyle oynanan oyunlar | Atari Pong |

---

## ⚡ Mülakat ve Teknik Değerlendirme Hazırlığı

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
