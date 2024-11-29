import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hilangkan karakter non-alfabet
    text = re.sub(r'\s+', ' ', text)  # Hilangkan spasi berlebih
    return text

# Load model klasifikasi aspek (Model 1)
aspect_model = joblib.load('random_forest_model_aspek.joblib')

# Load model klasifikasi sentimen (Model 2, 3, dan 4)
sentiment_models = {
    "fasilitas": joblib.load('model_random_forest_fasilitas.pkl'),
    "pelayanan": joblib.load('model_random_forest_pelayanan.pkl'),
    "masakan": joblib.load('model_random_forest_masakan.pkl')
}

# Load vectorizer masing-masing model
vectorizers = {
    "aspek": joblib.load('model_tfidf_aspek.pkl'),
    "fasilitas": joblib.load('tfidf_vectorizer_fasilitas.pkl'),
    "pelayanan": joblib.load('tfidf_vectorizer_pelayanan.pkl'),
    "masakan": joblib.load('tfidf_vectorizer_masakan.pkl')
}

# Streamlit App
def main():
    st.title("Sistem Prediksi Aspek dan Sentimen dengan Random Forest")
    st.markdown("""
    ### Sistem ini memprediksi:
    - **Aspek**: Fasilitas, Pelayanan, Masakan
    - **Sentimen**: Positif atau Negatif
    """)

    # Input teks dari user
    user_input = st.text_area("Masukkan Teks", "")

    if st.button("Prediksi"):
        # Preprocessing teks
        processed_text = preprocess_text(user_input)

        # Step 1: Prediksi aspek (Model 1)
        aspect_vectorized = vectorizers["aspek"].transform([processed_text])
        predicted_aspect = aspect_model.predict(aspect_vectorized)[0]

        # Step 2: Prediksi sentimen berdasarkan aspek (Model 2, 3, 4)
        if predicted_aspect in sentiment_models:
            sentiment_vectorizer = vectorizers[predicted_aspect]
            sentiment_model = sentiment_models[predicted_aspect]

            sentiment_vectorized = sentiment_vectorizer.transform([processed_text])
            predicted_sentiment = sentiment_model.predict(sentiment_vectorized)[0]

            # Tampilkan hasil
            st.write(f"**Aspek**: {predicted_aspect.capitalize()}")
            st.write(f"**Sentimen**: {predicted_sentiment.capitalize()}")
        else:
            st.error("Aspek tidak dikenali.")

if __name__ == "__main__":
    main()
