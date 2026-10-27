package com.smartwakeup.app.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build

/**
 * AlarmReceiver - AlarmManager tarafından hedef saatten 30 dakika önce tetiklenir.
 * Telefon kilitli veya uyku modunda olsa bile cihazı uyandırıp WakeUpService'i başlatır.
 */
class AlarmReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        val serviceIntent = Intent(context, WakeUpService::class.java).apply {
            putExtra("TARGET_TIMESTAMP", intent.getLongExtra("TARGET_TIMESTAMP", System.currentTimeMillis()))
            putExtra("TOTAL_SLEEP_HOURS", intent.getFloatExtra("TOTAL_SLEEP_HOURS", 7.0f))
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(serviceIntent)
        } else {
            context.startService(serviceIntent)
        }
    }
}
