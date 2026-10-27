package com.smartwakeup.app.ui

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.smartwakeup.app.service.WakeUpService

/**
 * AlarmActivity - Ekran kilitliyken bile tam ekran olarak açılan,
 * ekran parlaklığını ve rengini gün doğumu gibi kademeli açan alarm ekranı.
 */
class AlarmActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var tvTime: TextView
    private lateinit var tvAction: TextView
    private lateinit var btnDismiss: Button

    private val updateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val timeToTarget = intent?.getIntExtra("TIME_TO_TARGET", 0) ?: 0
            val actionName = intent?.getStringExtra("ACTION_NAME") ?: ""
            val lightLevel = intent?.getFloatExtra("LIGHT_LEVEL", 0f) ?: 0f

            tvTime.text = "${if (timeToTarget >= 0) "+$timeToTarget" else "$timeToTarget"} dk"
            tvAction.text = "Yapay Zeka: $actionName"
            tvStatus.text = "Işık: %${(lightLevel * 100).toInt()}"

            // Ekran Parlaklığını Ayarla
            val layoutParams = window.attributes
            layoutParams.screenBrightness = maxOf(0.1f, lightLevel)
            window.attributes = layoutParams

            // Gün Doğumu Rengi Geçişi (Karanlık Mor -> Sıcak Turuncu -> Gün Işığı)
            updateBackgroundColor(lightLevel)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Kilit ekranının üstünde ve ekranı açarak görünmesini sağla
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
            )
        }

        // Layout oluşturma (Programatik / Basit ve Hızlı)
        val layout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            gravity = android.view.Gravity.CENTER
            setBackgroundColor(Color.parseColor("#1A0933"))
            setPadding(40, 40, 40, 40)
        }

        val tvTitle = TextView(this).apply {
            text = "🌅 Akıllı Uyanma Zamanı"
            textSize = 28f
            setTextColor(Color.WHITE)
            gravity = android.view.Gravity.CENTER
        }
        layout.addView(tvTitle)

        tvTime = TextView(this).apply {
            text = "-30 dk"
            textSize = 48f
            setTextColor(Color.parseColor("#FFA726"))
            gravity = android.view.Gravity.CENTER
            setPadding(0, 30, 0, 10)
        }
        layout.addView(tvTime)

        tvAction = TextView(this).apply {
            text = "Yapay Zeka Hazırlanıyor..."
            textSize = 20f
            setTextColor(Color.LTGRAY)
            gravity = android.view.Gravity.CENTER
            setPadding(0, 0, 0, 20)
        }
        layout.addView(tvAction)

        tvStatus = TextView(this).apply {
            text = "Işık: %0"
            textSize = 18f
            setTextColor(Color.WHITE)
            gravity = android.view.Gravity.CENTER
            setPadding(0, 0, 0, 50)
        }
        layout.addView(tvStatus)

        btnDismiss = Button(this).apply {
            text = "🏃 KALKTIM! (Alarmı Kapat)"
            textSize = 20f
            setBackgroundColor(Color.parseColor("#4CAF50"))
            setTextColor(Color.WHITE)
            setPadding(30, 20, 30, 20)
            setOnClickListener {
                dismissAlarm()
            }
        }
        layout.addView(btnDismiss)

        setContentView(layout)

        // Güncellemeleri dinle
        val filter = IntentFilter(WakeUpService.BROADCAST_UPDATE)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(updateReceiver, filter, RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(updateReceiver, filter)
        }
    }

    private fun updateBackgroundColor(lightLevel: Float) {
        // %0 iken #1A0933 (Karanlık Mor) -> %100 iken #FFA726 (Gün Doğumu Turuncusu)
        val r = (0x1A + (0xFF - 0x1A) * lightLevel).toInt()
        val g = (0x09 + (0xA7 - 0x09) * lightLevel).toInt()
        val b = (0x33 + (0x26 - 0x33) * lightLevel).toInt()
        window.decorView.setBackgroundColor(Color.rgb(r, g, b))
    }

    private fun dismissAlarm() {
        val stopIntent = Intent(this, WakeUpService::class.java).apply {
            action = WakeUpService.ACTION_DISMISS
        }
        startService(stopIntent)
        finish()
    }

    override fun onDestroy() {
        try {
            unregisterReceiver(updateReceiver)
        } catch (e: Exception) {}
        super.onDestroy()
    }
}
