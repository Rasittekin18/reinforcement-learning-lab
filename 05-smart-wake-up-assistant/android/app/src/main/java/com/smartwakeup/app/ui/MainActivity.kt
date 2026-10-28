package com.smartwakeup.app.ui

import android.app.AlarmManager
import android.app.PendingIntent
import android.app.TimePickerDialog
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.smartwakeup.app.service.AlarmReceiver
import com.smartwakeup.app.service.WakeUpService
import java.util.*

/**
 * MainActivity - Kullanıcının hedef alarm saatini belirlediği,
 * AlarmManager ile sisteme kaydettiği ve hemen canlı test yapabildiği ana ekran.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var tvTargetTime: TextView
    private lateinit var tvStatus: TextView
    private var targetHour = 7
    private var targetMinute = 30

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // İzin kontrolü (Android 12+ Exact Alarm ve Android 13+ Bildirim)
        checkExactAlarmPermission()
        checkNotificationPermission()

        // Programatik Temiz Layout
        val layout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            gravity = android.view.Gravity.CENTER
            setPadding(50, 50, 50, 50)
            setBackgroundColor(android.graphics.Color.parseColor("#121212"))
        }

        val tvTitle = TextView(this).apply {
            text = "🌅 Akıllı Uyanma Asistanı"
            textSize = 26f
            setTextColor(android.graphics.Color.WHITE)
            gravity = android.view.Gravity.CENTER
            setPadding(0, 0, 0, 40)
        }
        layout.addView(tvTitle)

        tvTargetTime = TextView(this).apply {
            text = String.format("%02d:%02d", targetHour, targetMinute)
            textSize = 64f
            setTextColor(android.graphics.Color.parseColor("#FFA726"))
            gravity = android.view.Gravity.CENTER
            setOnClickListener { showTimePicker() }
        }
        layout.addView(tvTargetTime)

        val btnChangeTime = Button(this).apply {
            text = "Saati Değiştir"
            setOnClickListener { showTimePicker() }
        }
        layout.addView(btnChangeTime)

        tvStatus = TextView(this).apply {
            text = "Alarm kurulu değil.\n(Yapay zeka hedef saatten 30 dk önce ışığı kademeli açmaya başlar)"
            textSize = 15f
            setTextColor(android.graphics.Color.LTGRAY)
            gravity = android.view.Gravity.CENTER
            setPadding(0, 30, 0, 40)
        }
        layout.addView(tvStatus)

        val btnSetAlarm = Button(this).apply {
            text = "⏰ Alarmı Kur (DQN Aktif)"
            textSize = 18f
            setBackgroundColor(android.graphics.Color.parseColor("#FF9800"))
            setTextColor(android.graphics.Color.WHITE)
            setPadding(30, 20, 30, 20)
            setOnClickListener { scheduleSmartAlarm() }
        }
        layout.addView(btnSetAlarm)

        // Hızlı Test Butonu (Beklemeden canlı denemek için)
        val btnQuickTest = Button(this).apply {
            text = "⚡ Hemen Canlı Test Et (Şimdi Başlat)"
            textSize = 16f
            setBackgroundColor(android.graphics.Color.parseColor("#2196F3"))
            setTextColor(android.graphics.Color.WHITE)
            setPadding(20, 15, 20, 15)
            setOnClickListener {
                Toast.makeText(this@MainActivity, "Akıllı uyanma başlatılıyor...", Toast.LENGTH_SHORT).show()
                val testIntent = Intent(this@MainActivity, WakeUpService::class.java).apply {
                    putExtra("TOTAL_SLEEP_HOURS", 6.5f)
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(testIntent)
                } else {
                    startService(testIntent)
                }
            }
        }
        layout.addView(btnQuickTest)

        setContentView(layout)
    }

    private fun showTimePicker() {
        val picker = TimePickerDialog(this, { _, hourOfDay, minute ->
            targetHour = hourOfDay
            targetMinute = minute
            tvTargetTime.text = String.format("%02d:%02d", targetHour, targetMinute)
        }, targetHour, targetMinute, true)
        picker.show()
    }

    private fun scheduleSmartAlarm() {
        val alarmManager = getSystemService(Context.ALARM_SERVICE) as AlarmManager

        val calendar = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, targetHour)
            set(Calendar.MINUTE, targetMinute)
            set(Calendar.SECOND, 0)
            // Hedef saatten 30 dakika önce tetikle
            add(Calendar.MINUTE, -30)
            if (timeInMillis <= System.currentTimeMillis()) {
                add(Calendar.DAY_OF_YEAR, 1) // Gelecek güne kur
            }
        }

        val intent = Intent(this, AlarmReceiver::class.java).apply {
            putExtra("TARGET_TIMESTAMP", calendar.timeInMillis + (30 * 60 * 1000L))
            putExtra("TOTAL_SLEEP_HOURS", 7.0f)
        }
        val pendingIntent = PendingIntent.getBroadcast(
            this, 100, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            alarmManager.setExactAndAllowWhileIdle(
                AlarmManager.RTC_WAKEUP,
                calendar.timeInMillis,
                pendingIntent
            )
        } else {
            alarmManager.setExact(
                AlarmManager.RTC_WAKEUP,
                calendar.timeInMillis,
                pendingIntent
            )
        }

        val triggerStr = String.format("%02d:%02d", calendar.get(Calendar.HOUR_OF_DAY), calendar.get(Calendar.MINUTE))
        tvStatus.text = "✅ Alarm Kuruldu!\nYapay zeka saat $triggerStr'de ışığı kademeli açmaya başlayacak."
        Toast.makeText(this, "Akıllı alarm kuruldu!", Toast.LENGTH_LONG).show()
    }

    private fun checkExactAlarmPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val alarmManager = getSystemService(Context.ALARM_SERVICE) as AlarmManager
            if (!alarmManager.canScheduleExactAlarms()) {
                val intent = Intent(Settings.ACTION_REQUEST_SCHEDULE_EXACT_ALARM)
                startActivity(intent)
            }
        }
    }

    private fun checkNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 101)
            }
        }
    }
}
