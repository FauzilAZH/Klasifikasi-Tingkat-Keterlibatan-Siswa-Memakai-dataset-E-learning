import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman (Tanpa Icon)
st.set_page_config(
    page_title="Prediksi Keterlibatan Siswa",
    layout="wide"
)

# Kustomisasi CSS untuk tampilan yang lebih rapi, modern, dan profesional
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Judul Utama */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0F172A;
        text-align: center;
        margin-bottom: 5px;
        margin-top: -30px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #475569;
        text-align: center;
        margin-bottom: 40px;
    }

    /* Container Hasil Prediksi */
    .result-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        padding: 30px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .result-label {
        font-size: 1.1rem;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 5px;
    }

    .result-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: #38BDF8;
        letter-spacing: -0.025em;
    }

    /* Styling tombol prediksi */
    div.stButton > button:first-child {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        height: 50px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Memuat model dan transformer
@st.cache_resource
def load_models():
    model = joblib.load('best_model_Decision_Tree_C4.5_Entropy_80_20.pkl')
    scaler = joblib.load('scaler (1).pkl')
    encoders = joblib.load('label_encoders (1).pkl')
    return model, scaler, encoders

try:
    model, scaler, encoders = load_models()
except FileNotFoundError:
    st.error("File model atau pendukung tidak ditemukan! Pastikan file berada di folder yang sama.")
    st.stop()

# Header Aplikasi
st.markdown('<div class="main-title">Sistem Analisis Pembelajaran Online</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prediksi tingkat keterlibatan siswa berdasarkan metrik aktivitas pembelajaran</div>', unsafe_allow_html=True)

# Membagi layout menjadi kolom di tengah untuk input (agar tidak menggunakan sidebar dan terlihat lebih rapi)
col0, col1, col2, col3 = st.columns([1, 4, 4, 1])

with col1:
    total_clicks = st.number_input("Total Klik Modul", min_value=0.0, value=500.0, step=10.0, help="Jumlah total interaksi klik siswa.")
    avg_score = st.number_input("Rata-Rata Nilai (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0, help="Nilai rata-rata dari aktivitas dan tugas.")

with col2:
    risk_level = st.selectbox("Tingkat Risiko", encoders['risk_level'].classes_, help="Profil risiko siswa saat ini.")
    gender = st.selectbox("Jenis Kelamin", encoders['gender'].classes_, help="M = Male, F = Female")


# Kolom khusus untuk memposisikan tombol di tengah
_, col_btn, _ = st.columns([2.5, 3, 2.5])
with col_btn:
    st.write("") # spacing vertikal
    submit_button = st.button("Jalankan Prediksi Keterlibatan")

if submit_button:
    # Memproses input menjadi DataFrame dan menyesuaikan nama & urutan fitur
    input_df = pd.DataFrame({
        'total_clicks': [total_clicks],
        'risk_level': [risk_level],
        'avg_score': [avg_score],
        'gender': [gender]
    })
    
    # Label encoding input kategorikal
    input_df['risk_level'] = encoders['risk_level'].transform(input_df['risk_level'])
    input_df['gender'] = encoders['gender'].transform(input_df['gender'])
    
    # Standard scaling untuk normalisasi
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        # Menghindari bug atribut versi lama scikit-learn jika nama fitur tidak disimpan
        input_scaled = scaler.transform(input_df[['total_clicks', 'risk_level', 'avg_score', 'gender']].values)

    # Menghasilkan Prediksi
    prediction_encoded = model.predict(input_scaled)
    # Decode ke string original
    prediction = encoders['engagement_level'].inverse_transform(prediction_encoded)[0]
    
    # Menampilkan hasil di tengah
    _, col_res, _ = st.columns([1, 6, 1])
    with col_res:
        st.markdown(f"""
        <div class="result-container">
            <div class="result-label">Berdasarkan data yang dimasukkan, klasifikasi keterlibatan siswa adalah</div>
            <div class="result-value">{prediction.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Grafik Distribusi Probabilitas 
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0]
            st.markdown("<p style='text-align:center; font-weight:500; color:#475569;'>Distribusi Kepercayaan Model Terhadap Prediksi:</p>", unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({
                "Tingkat Keterlibatan": encoders['engagement_level'].classes_,
                "Probabilitas (%)": proba * 100
            })
            
            # Tampilkan sebagai chart sederhana (menghapus background st.bar_chart default)
            st.bar_chart(prob_df.set_index("Tingkat Keterlibatan"), height=250)
