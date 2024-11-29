import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os

# Fungsi Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Fungsi Memuat Model dengan Validasi
def load_model(file_path):
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        st.error(f"Error: File '{file_path}' tidak ditemukan.")
        return None

# Memuat Model Aspek dan Sentimen
aspect_model = load_model('random_forest_model_aspek.pkl')

sentiment_models = {
    "fasilitas": load_model('model_random_forest_fasilitas.pkl'),
    "pelayanan": load_model('model_random_forest_pelayanan.pkl'),
    "masakan": load_model('model_random_forest_masakan.pkl')
}

# Aplikasi Streamlit
def main():
    st.title("Sistem Prediksi Aspek dan Sentimen dengan Random Forest")
    st.markdown("### Sistem ini memprediksi:\n- **Aspek**: Fasilitas, Pelayanan, Masakan\n- **Sentimen**: Positif atau Negatif")
    
    # Input dari pengguna
    user_input = st.text_area("Masukkan Teks", "")
    
    # Tombol Prediksi
    if st.button("Prediksi"):
        if aspect_model is None or any(model is None for model in sentiment_models.values()):
            st.error("Sistem belum siap, pastikan semua model tersedia.")
            return
        
        # Preprocessing
        processed_text = preprocess_text(user_input)
        
        # Prediksi Aspek
        predicted_aspect = aspect_model.predict([processed_text])[0]
        
        # Prediksi Sentimen berdasarkan Aspek
        if predicted_aspect in sentiment_models:
            sentiment_model = sentiment_models[predicted_aspect]
            predicted_sentiment = sentiment_model.predict([processed_text])[0]
            st.write(f"**Aspek**: {predicted_aspect.capitalize()}")
            st.write(f"**Sentimen**: {predicted_sentiment.capitalize()}")
        else:
            st.error("Aspek tidak dikenali.")

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
