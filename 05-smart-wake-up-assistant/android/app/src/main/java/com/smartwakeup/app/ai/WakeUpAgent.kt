package com.smartwakeup.app.ai

import android.content.Context
import org.json.JSONObject
import java.io.InputStreamReader

/**
 * WakeUpAgent - Eğitilmiş Deep Q-Network (DQN) Mobil İnference Motoru
 * Harici hiçbir ağır yapay zeka kütüphanesine (PyTorch/TensorFlow) ihtiyaç duymadan,
 * saf Kotlin matris işlemleriyle 0.05 milisaniyede eylem seçer.
 */
class WakeUpAgent(context: Context) {

    // Katman 1: 7 giriş -> 64 gizli düğüm (ReLU)
    private val w1 = Array(64) { FloatArray(7) }
    private val b1 = FloatArray(64)

    // Katman 2: 64 gizli düğüm -> 64 gizli düğüm (ReLU)
    private val w2 = Array(64) { FloatArray(64) }
    private val b2 = FloatArray(64)

    // Çıkış Katmanı: 64 gizli düğüm -> 5 eylem Q değeri
    private val w3 = Array(5) { FloatArray(64) }
    private val b3 = FloatArray(5)

    companion object {
        const val ACTION_DO_NOTHING = 0
        const val ACTION_INCREASE_LIGHT = 1
        const val ACTION_SOFT_SOUND = 2
        const val ACTION_ALARM_SOUND = 3
        const val ACTION_AGGRESSIVE_ALERT = 4

        val ACTION_NAMES = arrayOf(
            "Bekle (Uykuyu Bölme)",
            "Işığı Artır (Gün Doğumu)",
            "Hafif Melodi / Doğa Sesi",
            "Standart Alarm Çal",
            "Agresif Mod (Son Çare)"
        )
    }

    init {
        loadWeightsFromAssets(context)
    }

    /**
     * assets/model_weights.json dosyasından ağırlıkları yükler.
     */
    private fun loadWeightsFromAssets(context: Context) {
        val inputStream = context.assets.open("model_weights.json")
        val jsonString = InputStreamReader(inputStream).readText()
        val json = JSONObject(jsonString)

        // w1 & b1 (net.0)
        val jsonW1 = json.getJSONArray("net.0.weight")
        for (i in 0 until 64) {
            val row = jsonW1.getJSONArray(i)
            for (j in 0 until 7) {
                w1[i][j] = row.getDouble(j).toFloat()
            }
        }
        val jsonB1 = json.getJSONArray("net.0.bias")
        for (i in 0 until 64) {
            b1[i] = jsonB1.getDouble(i).toFloat()
        }

        // w2 & b2 (net.2)
        val jsonW2 = json.getJSONArray("net.2.weight")
        for (i in 0 until 64) {
            val row = jsonW2.getJSONArray(i)
            for (j in 0 until 64) {
                w2[i][j] = row.getDouble(j).toFloat()
            }
        }
        val jsonB2 = json.getJSONArray("net.2.bias")
        for (i in 0 until 64) {
            b2[i] = jsonB2.getDouble(i).toFloat()
        }

        // w3 & b3 (net.4)
        val jsonW3 = json.getJSONArray("net.4.weight")
        for (i in 0 until 5) {
            val row = jsonW3.getJSONArray(i)
            for (j in 0 until 64) {
                w3[i][j] = row.getDouble(j).toFloat()
            }
        }
        val jsonB3 = json.getJSONArray("net.4.bias")
        for (i in 0 until 5) {
            b3[i] = jsonB3.getDouble(i).toFloat()
        }
    }

    /**
     * Durum Vektörünü (7 float) alıp en yüksek Q değerine sahip en iyi eylemi (0-4) seçer.
     * State Vektörü:
     * [0] norm_time_to_target (-1.0 ile 1.0 arası)
     * [1] sleep_stage (0: Derin, 1: Hafif, 2: Yarı Uyanık, 3: Kalktı)
     * [2] sleep_inertia (0.0 - 1.0)
     * [3] light_level (0.0 - 1.0)
     * [4] sound_level (0.0 - 1.0)
     * [5] snooze_count (0 - 5)
     * [6] total_sleep_hours (örn. 7.0)
     */
    fun predict(state: FloatArray): Int {
        // 1. Gizli Katman: h1 = ReLU(W1 * x + b1)
        val h1 = FloatArray(64)
        for (i in 0 until 64) {
            var sum = b1[i]
            for (j in 0 until 7) {
                sum += w1[i][j] * state[j]
            }
            h1[i] = if (sum > 0f) sum else 0f
        }

        // 2. Gizli Katman: h2 = ReLU(W2 * h1 + b2)
        val h2 = FloatArray(64)
        for (i in 0 until 64) {
            var sum = b2[i]
            for (j in 0 until 64) {
                sum += w2[i][j] * h1[j]
            }
            h2[i] = if (sum > 0f) sum else 0f
        }

        // Çıkış Katmanı: q = W3 * h2 + b3
        var bestAction = 0
        var maxQ = -Float.MAX_VALUE
        for (i in 0 until 5) {
            var sum = b3[i]
            for (j in 0 until 64) {
                sum += w3[i][j] * h2[j]
            }
            if (sum > maxQ) {
                maxQ = sum
                bestAction = i
            }
        }

        return bestAction
    }
}
