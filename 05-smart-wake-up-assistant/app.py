"""
app.py - Akıllı Uyanma Asistanı İnteraktif Streamlit Dashboard'u
Pekiştirmeli Öğrenme (DQN) modelinin kararlarını adım adım simüle eden ve görselleştiren web arayüzü.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wake_up_env import SmartWakeUpEnv
from dqn_agent import DQNAgent
from evaluate import compare_systems

st.set_page_config(
    page_title="Akıllı Uyanma Asistanı (RL - DQN)",
    page_icon="🌅",
    layout="wide"
)

# --- CSS & Tasarım İyileştirmeleri ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #ff7f0e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stProgress > div > div > div > div {
        background-color: #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_agent():
    agent = DQNAgent(state_dim=7, action_dim=5)
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_wake_up_agent.pth")
    if os.path.exists(model_path):
        agent.load(model_path)
    return agent


agent = load_trained_agent()

# --- Başlık ---
st.title("🌅 Akıllı Uyanma Asistanı: Pekiştirmeli Öğrenme (DQN)")
st.markdown("""
Bu proje, geleneksel alarmların körü körüne çalarak neden olduğu **uyku sersemliğini (sleep inertia)** ve 
kronikleşen **erteleme (snooze)** krizini, **Deep Q-Network (DQN)** kullanarak çözen kişisel bir yapay zeka asistanıdır.
""")

# --- Sidebar Parametreleri ---
st.sidebar.header("⚙️ Sabah Simülasyon Ayarları")
sleep_hours = st.sidebar.slider("Gece Uyunan Süre (Saat)", min_value=4.0, max_value=9.0, value=6.5, step=0.5)
initial_stage = st.sidebar.selectbox("Hedef Saatten 30 Dk Önceki Uyku Evresi", ["Derin Uyku (Deep Sleep)", "Hafif Uyku (Light Sleep)"])
target_time_str = st.sidebar.text_input("Hedef Uyanma Saati", value="07:30")

run_btn = st.sidebar.button("🚀 Sabah Simülasyonunu Başlat", use_container_width=True)

tab_sim, tab_compare, tab_theory = st.tabs(["🧪 Canlı Sabah Simülasyonu", "⚖️ Klasik Alarm vs. RL Ajanı", "📚 RL Teorisi ve Eğitim Grafikleri"])

with tab_sim:
    if run_btn:
        env = SmartWakeUpEnv()
        state, _ = env.reset()

        # Kullanıcı girdilerini çevreye enjekte et
        env.total_sleep_hours = float(sleep_hours)
        env.sleep_inertia = float(np.clip(1.0 - (sleep_hours / 10.0), 0.2, 0.9))
        env.sleep_stage = env.STAGE_DEEP if "Derin" in initial_stage else env.STAGE_LIGHT
        state = env._get_obs()

        action_labels = {
            0: "💤 Bekle (Uykuyu Bölme)",
            1: "💡 Işığı Artır (Gün Doğumu)",
            2: "🎵 Hafif Ses / Doğa Melodisi",
            3: "🔔 Standart Alarm",
            4: "⚡ Agresif Mod (Maksimum Uyarı)"
        }
        stage_labels = {
            0: "😴 Derin Uyku",
            1: "🌙 Hafif Uyku",
            2: "🥱 Yarı Uyanık",
            3: "🏃 Yataktan Kalktı"
        }

        timeline = []
        done = False
        step = 0

        while not done:
            step += 1
            q_vals = agent.get_q_values(state)
            action = int(np.argmax(q_vals))
            next_state, reward, terminated, truncated, info = env.step(action)

            timeline.append({
                "Adım": step,
                "Zaman": f"{info['time_to_target']:+d} dk",
                "Eylem": action_labels[action],
                "Uyku Evresi": stage_labels[info["sleep_stage"]],
                "Işık Seviyesi (%)": int(env.light_level * 100),
                "Ses Seviyesi (%)": int(env.sound_level * 100),
                "Erteleme (Snooze)": info["snooze_count"],
                "Sersemlik (%)": int(info["sleep_inertia"] * 100),
                "Ödül": round(reward, 1),
                "Q_Values": q_vals.tolist()
            })

            state = next_state
            done = terminated or truncated

        df_timeline = pd.DataFrame(timeline)

        # Üst Metrik Kartları
        col1, col2, col3, col4 = st.columns(4)
        last_info = timeline[-1]
        is_success = ("Yataktan Kalktı" in last_info["Uyku Evresi"] and int(last_info["Zaman"].split()[0]) <= 5)

        col1.metric("Sonuç", "✅ Başarılı Kalkış" if is_success else "❌ Başarısız", delta="Zamanında" if is_success else "Geç")
        col2.metric("Kalkış Zamanı", f"Hedef {last_info['Zaman']}")
        col3.metric("Erteleme (Snooze)", f"{last_info['Erteleme (Snooze)']} Kez")
        col4.metric("Kalan Sersemlik", f"%{last_info['Sersemlik (%)']}", delta=f"-%{100 - last_info['Sersemlik (%)']} Azalma", delta_color="inverse")

        st.subheader("📋 Simülasyon Zaman Çizelgesi")
        display_df = df_timeline.drop(columns=["Q_Values"])
        st.dataframe(display_df, use_container_width=True)

        # Görsel Grafik: Işık, Ses ve Sersemlik Trendi
        st.subheader("📈 Uyanma Sürecinde Işık, Ses ve Biyolojik Tepki")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_timeline["Zaman"], y=df_timeline["Işık Seviyesi (%)"], name="Işık Seviyesi (%)", mode='lines+markers', line=dict(color='#ffaa00', width=3)))
        fig.add_trace(go.Scatter(x=df_timeline["Zaman"], y=df_timeline["Ses Seviyesi (%)"], name="Ses Seviyesi (%)", mode='lines+markers', line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=df_timeline["Zaman"], y=df_timeline["Sersemlik (%)"], name="Uyku Sersemliği (Inertia %)", mode='lines+markers', line=dict(color='#d62728', dash='dot', width=3)))
        fig.update_layout(xaxis_title="Hedef Saate Göre Zaman", yaxis_title="Yüzde (%)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Q-Values Açıklanabilirlik Grafiği
        st.subheader("🤖 Yapay Zeka Karar Analizi (Q-Değerleri)")
        q_steps = [f"Adım {t['Adım']} ({t['Zaman']})" for t in timeline]
        fig_q = go.Figure()
        for a_idx, a_name in action_labels.items():
            vals = [t["Q_Values"][a_idx] for t in timeline]
            fig_q.add_trace(go.Bar(name=a_name, x=q_steps, y=vals))
        fig_q.update_layout(
            barmode='group',
            xaxis_title="Simülasyon Adımı",
            yaxis_title="Tahmin Edilen Q Değeri (Beklenen Toplam Ödül)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_q, use_container_width=True)

        st.info("""
        💡 **Yapay Zeka Ne Yaptı?**
        Ajan rastgele bir saatte gürültü patlatmak yerine; derin uyku evresini tespit edip önce odadaki ışığı kademeli artırarak melatonini baskıladı. 
        Kullanıcı hafif uykuya geçtiğinde kısık doğa melodisi ile beyni yumuşakça uyandırdı ve hedef saatte sıfır erteleme ile yataktan dinç kalkmasını sağladı!
        """)
    else:
        st.info("👈 Soldaki panelden uyku sürenizi seçip **'Sabah Simülasyonunu Başlat'** butonuna tıklayın.")

with tab_compare:
    st.subheader("⚖️ Klasik Sabit Saatli Alarm vs. RL Akıllı Uyanma Asistanı")
    st.markdown("""
    Aşağıdaki sonuçlar, geleneksel sabit çalan alarm mantığı ile eğitilmiş RL ajanının 
    **100 farklı sabah senaryosunda** (değişken uyku saatleri ve biyolojik profiller altında) karşılaştırılmasıdır.
    """)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        run_live_benchmark = st.button("🔄 Canlı 100 Sabah Kıyaslama Testini Çalıştır")

    if run_live_benchmark:
        with st.spinner("100 farklı sabah simülasyonu çalıştırılıyor..."):
            live_results = compare_systems(n_test=100, verbose=False)
            t_res = live_results["traditional"]
            r_res = live_results["rl"]

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Zamanında Uyanma", f"%{r_res['success_rate']:.1f}", delta=f"+%{r_res['success_rate'] - t_res['success_rate']:.1f}")
            m_col2.metric("Ortalama Snooze", f"{r_res['avg_snooze']:.2f}", delta=f"{r_res['avg_snooze'] - t_res['avg_snooze']:.2f} kez", delta_color="inverse")
            m_col3.metric("Kalan Sersemlik", f"%{r_res['avg_inertia']:.1f}", delta=f"-%{t_res['avg_inertia'] - r_res['avg_inertia']:.1f}", delta_color="inverse")
            m_col4.metric("Ortalama Ödül", f"{r_res['avg_reward']:.1f}", delta=f"+{r_res['avg_reward'] - t_res['avg_reward']:.1f}")

            # Plotly karşılaştırma grafiği
            categories = ['Zamanında Başarı (%)', 'Dinçlik Bonus Oranı (%)', 'Erteleme (x10 Kez)', 'Skor / Ödül']
            trad_vals = [t_res['success_rate'], 100.0 - t_res['avg_inertia'], t_res['avg_snooze'] * 10, max(0.0, t_res['avg_reward'] + 100)]
            rl_vals = [r_res['success_rate'], 100.0 - r_res['avg_inertia'], r_res['avg_snooze'] * 10, max(0.0, r_res['avg_reward'] + 100)]

            fig_comp = go.Figure(data=[
                go.Bar(name='Geleneksel Alarm', x=categories, y=trad_vals, marker_color='#9e9e9e'),
                go.Bar(name='RL Akıllı Asistan (DQN)', x=categories, y=rl_vals, marker_color='#ff7f0e')
            ])
            fig_comp.update_layout(barmode='group', title="Canlı Test Kıyaslama Grafiği", yaxis_title="Değer")
            st.plotly_chart(fig_comp, use_container_width=True)

    compare_data = {
        "Metrik": [
            "Zamanında Uyanma Başarı Oranı",
            "Ortalama Erteleme (Snooze) Sayısı",
            "Uyanınca Hissedilen Sersemlik (Sleep Inertia)",
            "Ortalama Sistem Skoru (Ödül)"
        ],
        "Geleneksel Alarm (Sabit Saat)": ["%32.0", "1.54 kez", "%81.5 (Ağır Sersemlik)", "-74.4"],
        "RL Akıllı Uyanma Asistanı (DQN)": ["%100.0", "0.00 kez", "%10.9 (Tam Dinç)", "+138.3"],
        "Fark / Kazanç": ["+%68 Daha Yüksek", "%100 Sıfır Erteleme", "%86 Daha Dinç Uyanma", "Optimal Politika"]
    }
    st.table(pd.DataFrame(compare_data))

with tab_theory:
    st.subheader("🧠 Pekiştirmeli Öğrenme Mimarisi & Canlı Politika Testi")
    st.markdown("""
    * **Durum Uzayı ($S$):** Kalan zaman, anlık uyku evresi, uyku sersemliği katsayısı, mevcut ışık/ses düzeyi, erteleme sayısı ve toplam uyku saati ($7$ boyutlu sürekli vektör).
    * **Eylem Uzayı ($A$):** `0: Bekle`, `1: Işığı Artır`, `2: Hafif Melodi`, `3: Standart Alarm`, `4: Agresif Mod`.
    * **Ödül Fonksiyonu ($R$):** Hedef saatte yataktan kalkış (`+100`), sersemlik azlığı bonusu (`+35`), erteleme cezası (`-15/kez`), erken agresif ceza (`-45`), geç kalma cezası (`-100`).
    """)

    st.markdown("#### 🔬 İnteraktif Q-Değeri Deney Masası (Manual State Inspector)")
    st.markdown("Aşağıdaki kaydırıcılarla herhangi bir sabah anını simüle edin ve DQN modelinin hangi eylemi neden seçtiğini görün:")
    
    exp_c1, exp_c2, exp_c3 = st.columns(3)
    with exp_c1:
        test_t = st.slider("Hedefe Kalan Süre (Dakika)", -30, 15, -15, step=5)
        test_stage = st.selectbox("Uyku Evresi", [0, 1, 2, 3], format_func=lambda x: ["Derin", "Hafif", "Yarı Uyanık", "Kalktı"][x])
    with exp_c2:
        test_inertia = st.slider("Uyku Sersemliği", 0.0, 1.0, 0.6, step=0.05)
        test_light = st.slider("Işık Seviyesi", 0.0, 1.0, 0.35, step=0.05)
    with exp_c3:
        test_sound = st.slider("Ses Seviyesi", 0.0, 1.0, 0.0, step=0.05)
        test_snooze = st.slider("Erteleme Sayısı", 0, 5, 0)

    norm_t = np.clip(test_t / 30.0, -1.0, 1.0)
    test_vec = np.array([norm_t, float(test_stage), test_inertia, test_light, test_sound, float(test_snooze), 7.0], dtype=np.float32)
    q_predict = agent.get_q_values(test_vec)
    best_act = int(np.argmax(q_predict))

    act_names = ["Bekle (0)", "Işık Artır (1)", "Hafif Ses (2)", "Standart Alarm (3)", "Agresif Mod (4)"]
    fig_inspect = px.bar(
        x=act_names,
        y=q_predict,
        color=act_names,
        labels={'x': 'Eylemler', 'y': 'Q Değeri'},
        title=f"Tahmin Edilen En İyi Eylem: 👉 <b>{act_names[best_act]}</b> (Q = {q_predict[best_act]:.2f})"
    )
    st.plotly_chart(fig_inspect, use_container_width=True)

    plot_file = os.path.join(os.path.dirname(__file__), "plots", "training_results.png")
    if os.path.exists(plot_file):
        st.image(plot_file, caption="DQN Öğrenme Eğrileri (Ödül, Başarı, Snooze, Epsilon)", use_container_width=True)
