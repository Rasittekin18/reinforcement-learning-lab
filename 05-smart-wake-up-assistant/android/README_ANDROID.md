# 📱 Akıllı Uyanma Asistanı - Android Uygulama Dokümantasyonu (Kotlin)

Bu Android uygulaması, Python'da eğitilen **Deep Q-Network (DQN)** yapay zeka modelini telefonun içinde **internetsiz, bilgisayarsız ve arka planda** çalıştıran tam teşekküllü bir akıllı alarmdır.

---

## 🏗️ Mimari ve Bileşenler

```text
android/
├── app/
│   ├── src/main/
│   │   ├── assets/
│   │   │   └── model_weights.json         # 44 KB'lık eğitilmiş yapay zeka ağırlıkları
│   │   ├── java/com/smartwakeup/app/
│   │   │   ├── ai/
│   │   │   │   └── WakeUpAgent.kt        # Saf Kotlin DQN çıkarım (Inference) motoru (Sıfır kütüphane bağımlılığı)
│   │   │   ├── service/
│   │   │   │   ├── AlarmReceiver.kt      # AlarmManager tarafından hedefe 30 dk kala telefonu uyandıran alıcı
│   │   │   │   └── WakeUpService.kt      # Kilit ekranında ışığı ve sesi kademeli yöneten Foreground Servisi
│   │   │   └── ui/
│   │   │       ├── MainActivity.kt       # Alarm saatini seçtiğin ve hemen canlı test edebileceğin ana ekran
│   │   │       └── AlarmActivity.kt      # Kilit ekranı üstünde gün doğumu ışığı açan ve "Kalktım" butonu sunan ekran
│   │   └── AndroidManifest.xml           # WakeLock, Exact Alarm ve Foreground Service izinleri
│   └── build.gradle.kts
```

---

## 🧠 Yapay Zeka Telefonda Nasıl Çalışıyor? (`WakeUpAgent.kt`)
* Harici hiçbir ağır PyTorch / TensorFlow kütüphanesine ihtiyaç duyulmaz.
* `assets/model_weights.json` içindeki 4.997 adet ağırlık katsayısı okunur.
* Kotlin matris çarpımları ile her eylem kararı **0.05 milisaniyede** (salisenin binde birinde) ve neredeyse **sıfır batarya tüketimiyle** verilir.

---

## 🌅 Gün Doğumu ve Ses Kontrolü
1. **Hedefe 30 Dk Kala:** Telefon uyku modundan çıkar, ekranı uyandırır.
2. **Işık Artışı:** `WindowManager.LayoutParams.screenBrightness` değeri %35 $\rightarrow$ %70 $\rightarrow$ %100 yapılır. Ekran rengi karanlık mordan sıcak gün doğumu turuncusuna döner.
3. **Kademeli Ses:** `RingtoneManager` ile önce kısık melodi, hedef saatte ise standart alarm çalar.
4. **"Kalktım!" Butonu:** Kullanıcı butona bastığında servis sonlanır, ekran parlaklığı normale döner.

---

## 🚀 Android Studio ile Çalıştırma Adımları

1. **Android Studio**'yu açın.
2. `Open` diyerek bu klasörü (`6-Proje-Akilli-Uyanma-Asistani-RL/android`) seçin.
3. Gradle senkronizasyonu tamamlandıktan sonra telefonunuzu USB ile bağlayın (veya emülatör seçin).
4. Yeşil **`Run 'app'`** butonuna basarak uygulamayı telefonunuza yükleyin.
5. Uygulama açıldığında:
   * **"⚡ Hemen Canlı Test Et"** butonuna basarak ekranın nasıl aydınlandığını ve sesin nasıl devreye girdiğini anında deneyebilirsiniz.
   * Veya yarın sabah için alarm saatinizi seçip **"Alarmı Kur"** butonuna basıp bilgisayarı kapatıp uyuyabilirsiniz!
