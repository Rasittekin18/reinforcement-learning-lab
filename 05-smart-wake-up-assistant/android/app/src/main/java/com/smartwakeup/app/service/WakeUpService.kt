package com.smartwakeup.app.service

import android.app.*
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.Ringtone
import android.media.RingtoneManager
import android.os.*
import androidx.core.app.NotificationCompat
import com.smartwakeup.app.ai.WakeUpAgent
import com.smartwakeup.app.ui.AlarmActivity
import kotlin.math.max
import kotlin.math.min

/**
 * WakeUpService - Hedef saatten 30 dakika önce başlayan ve uyanma sürecini
 * Deep Q-Network ajanı ile yöneten Foreground Servisi.
 */
class WakeUpService : Service() {

    private lateinit var agent: WakeUpAgent
    private var handler: Handler? = null
    private var runnable: Runnable? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var ringtone: Ringtone? = null

    // Simülasyon / Durum Değişkenleri
    private var timeToTarget = -30
    private var sleepStage = 0 // 0: Derin, 1: Hafif, 2: Yarı Uyanık, 3: Kalktı
    private var sleepInertia = 0.5f
    private var lightLevel = 0.0f
    private var soundLevel = 0.0f
    private var snoozeCount = 0
    private var totalSleepHours = 7.0f

    companion object {
        const val CHANNEL_ID = "smart_wake_up_channel"
        const val NOTIFICATION_ID = 1001
        const val ACTION_DISMISS = "com.smartwakeup.app.ACTION_DISMISS"
        const val ACTION_SNOOZE = "com.smartwakeup.app.ACTION_SNOOZE"
        const val BROADCAST_UPDATE = "com.smartwakeup.app.BROADCAST_UPDATE"
    }

    override fun onCreate() {
        super.onCreate()
        agent = WakeUpAgent(this)
        createNotificationChannel()

        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP,
            "SmartWakeUp::ServiceWakeLock"
        )
        wakeLock?.acquire(45 * 60 * 1000L) // 45 dakika maksimum
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_DISMISS) {
            stopWakeUpProcess()
            return START_NOT_STICKY
        }

        totalSleepHours = intent?.getFloatExtra("TOTAL_SLEEP_HOURS", 7.0f) ?: 7.0f
        sleepInertia = max(0.2f, min(0.85f, 1.0f - (totalSleepHours / 10.0f)))

        startForeground(NOTIFICATION_ID, buildNotification("Akıllı uyanma süreci başlatıldı..."))

        // Tam ekran alarm arayüzünü aç
        val alarmIntent = Intent(this, AlarmActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }
        startActivity(alarmIntent)

        startAgentLoop()

        return START_STICKY
    }

    private fun startAgentLoop() {
        handler = Handler(Looper.getMainLooper())
        runnable = object : Runnable {
            override fun run() {
                val action = stepAgent()

                // Ekran arayüzüne güncel durumu yayınla
                val updateIntent = Intent(BROADCAST_UPDATE).apply {
                    putExtra("TIME_TO_TARGET", timeToTarget)
                    putExtra("ACTION_NAME", WakeUpAgent.ACTION_NAMES[action])
                    putExtra("LIGHT_LEVEL", lightLevel)
                    putExtra("SOUND_LEVEL", soundLevel)
                    putExtra("INERTIA", sleepInertia)
                }
                sendBroadcast(updateIntent)

                // Kullanıcı kalktıysa veya süre bittiyse durdur
                if (sleepStage == 3 || timeToTarget >= 15) {
                    val statusText = if (sleepStage == 3) {
                        "Tebrikler! Yataktan kalktınız. Dinç bir gün dileriz!"
                    } else {
                        "Süre doldu. Lütfen yataktan kalkınız!"
                    }
                    val manager = getSystemService(NotificationManager::class.java)
                    manager?.notify(NOTIFICATION_ID, buildNotification(statusText))
                    // 60 saniye sonra alarm sesini durdurup servisi otomatik temizle
                    handler?.postDelayed({
                        stopWakeUpProcess()
                    }, 60 * 1000L)
                } else {
                    // Bir sonraki adım (Gerçekte 5 dakika, testte hızlı ilerletilebilir)
                    handler?.postDelayed(this, 5 * 60 * 1000L)
                }
            }
        }
        handler?.post(runnable!!)
    }

    private fun stepAgent(): Int {
        val normTime = max(-1.0f, min(1.0f, timeToTarget / 30.0f))
        val state = floatArrayOf(
            normTime,
            sleepStage.toFloat(),
            sleepInertia,
            lightLevel,
            soundLevel,
            snoozeCount.toFloat(),
            totalSleepHours
        )

        val action = agent.predict(state)

        // Eylemi Gerçekleştir
        when (action) {
            WakeUpAgent.ACTION_INCREASE_LIGHT -> {
                lightLevel = min(1.0f, lightLevel + 0.35f)
                if (sleepStage == 0 && lightLevel >= 0.5f) {
                    sleepStage = 1 // Hafif uykuya geçiş
                    sleepInertia = max(0.1f, sleepInertia - 0.2f)
                }
            }
            WakeUpAgent.ACTION_SOFT_SOUND -> {
                soundLevel = 0.35f
                if (sleepStage == 1) {
                    sleepStage = 2 // Yarı uyanıklık
                    sleepInertia = max(0.1f, sleepInertia - 0.15f)
                }
                playTone(0.35f)
            }
            WakeUpAgent.ACTION_ALARM_SOUND -> {
                soundLevel = 0.75f
                sleepStage = 3 // Yataktan kalktı
                playTone(0.85f)
            }
            WakeUpAgent.ACTION_AGGRESSIVE_ALERT -> {
                lightLevel = 1.0f
                soundLevel = 1.0f
                sleepStage = 3
                playTone(1.0f)
            }
        }

        updateNotification(action)
        timeToTarget += 5
        return action
    }

    private fun playTone(volume: Float) {
        if (ringtone == null) {
            val alertUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
                ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
            ringtone = RingtoneManager.getRingtone(applicationContext, alertUri)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            ringtone?.volume = volume
        }
        if (ringtone?.isPlaying == false) {
            ringtone?.play()
        }
    }

    private fun stopWakeUpProcess() {
        handler?.removeCallbacks(runnable ?: Runnable {})
        ringtone?.stop()
        wakeLock?.let { if (it.isHeld) it.release() }
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Akıllı Uyanma Servisi",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Yapay zeka uyanma sürecini yönetiyor"
                setSound(null, null)
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String): Notification {
        val dismissIntent = Intent(this, WakeUpService::class.java).apply {
            action = ACTION_DISMISS
        }
        val dismissPendingIntent = PendingIntent.getService(
            this, 0, dismissIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("🌅 Akıllı Uyanma Asistanı")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_lock_idle_alarm)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel, "Kalktım!", dismissPendingIntent)
            .build()
    }

    private fun updateNotification(action: Int) {
        val text = "Eylem: ${WakeUpAgent.ACTION_NAMES[action]} | Işık: %${(lightLevel*100).toInt()}"
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, buildNotification(text))
    }

    override fun onDestroy() {
        stopWakeUpProcess()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
