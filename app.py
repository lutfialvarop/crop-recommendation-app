import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Path file
MODEL_PATH = "crop_model_pipeline.pkl"
ENCODER_PATH = "label_encoder_classes.pkl"
SCALER_PATH = "scaler.pkl"

# Fungsi untuk memuat model dan encoder
@st.cache_resource
def load_resources():
    # Cek file
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH) or not os.path.exists(SCALER_PATH):
        st.error(f"File model '{MODEL_PATH}', '{ENCODER_PATH}', atau '{SCALER_PATH}' tidak ditemukan.")
        st.error("Pastikan Anda telah menjalankan 'train_model.py' terlebih dahulu.")
        return None, None, None
        
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        with open(ENCODER_PATH, 'rb') as f:
            classes = pickle.load(f)
            
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # Patch untuk monotonic_cst attribute error (kompatibilitas scikit-learn)
        def _patch_monotonic_attr(estimator):
            if hasattr(estimator, "estimators_"):  # Ensemble models
                for sub in estimator.estimators_:
                    if not hasattr(sub, "monotonic_cst"):
                        setattr(sub, "monotonic_cst", None)
            if hasattr(estimator, "tree_") and not hasattr(estimator, "monotonic_cst"):
                setattr(estimator, "monotonic_cst", None)
        
        # Apply patch ke model dalam pipeline
        try:
            model_in_pipeline = model.named_steps.get("model", None)
            if model_in_pipeline is not None:
                _patch_monotonic_attr(model_in_pipeline)
        except:
            pass
            
        return model, classes, scaler
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat resources: {e}")
        return None, None, None

# Memuat resources
model, classes, scaler = load_resources()

# Judul Aplikasi
st.title("🌱 Aplikasi Rekomendasi Tanaman")
st.markdown("Masukkan parameter tanah dan lingkungan untuk mendapatkan rekomendasi 3 tanaman yang paling cocok.")

# Sidebar untuk input
st.sidebar.header("Input Parameter")
def user_input_features():
    N = st.sidebar.slider('Nitrogen (N)', min_value=0, max_value=150, value=90)
    P = st.sidebar.slider('Phosphorus (P)', min_value=0, max_value=150, value=42)
    K = st.sidebar.slider('Potassium (K)', min_value=0, max_value=210, value=43)
    temperature = st.sidebar.slider('Suhu (°C)', min_value=8.0, max_value=45.0, value=20.8, step=0.1)
    humidity = st.sidebar.slider('Kelembaban (%)', min_value=14.0, max_value=100.0, value=82.0, step=0.1)
    ph = st.sidebar.slider('pH Tanah', min_value=3.5, max_value=9.5, value=6.5, step=0.1)
    rainfall = st.sidebar.slider('Curah Hujan (mm)', min_value=20.0, max_value=300.0, value=202.9, step=0.1)
    
    # Kumpulkan data input
    data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    
    # Ubah ke DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Tampilkan input di main area
st.subheader("Parameter Input:")
st.dataframe(input_df)

if st.sidebar.button("Dapatkan Rekomendasi"):
    if model is not None and classes is not None:
        try:
            # Dapatkan probabilitas prediksi
            probabilities = model.predict_proba(input_df)
            
            # Ambil probabilitas untuk satu input
            probs = probabilities[0]
            
            # Dapatkan 3 indeks teratas (dengan probabilitas tertinggi)
            # np.argsort sort descending
            top_3_indices = np.argsort(probs)[-3:][::-1]
            
            # Dapatkan nama tanaman dan probabilitasnya
            top_3_crops = [classes[i] for i in top_3_indices]
            top_3_probs = [probs[i] for i in top_3_indices]
            
            # Tampilkan hasil
            st.subheader("🌟 Top 3 Rekomendasi Tanaman:")
            for i in range(3):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"## {i+1}.")
                with col2:
                    st.success(f"**{top_3_crops[i].capitalize()}**")
                    st.progress(int(top_3_probs[i] * 100))
                    st.markdown(f"**Probabilitas: {top_3_probs[i]*100:.2f}%**")
                    st.markdown("---")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
    else:
        st.warning("Model tidak dapat dimuat. Silakan cek error di atas.")
else:
    st.info("Klik tombol 'Dapatkan Rekomendasi' di sidebar setelah mengatur parameter.")