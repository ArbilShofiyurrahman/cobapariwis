import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

aspect_model = joblib.load('random_forest_model_aspek.pkl')

sentiment_models = {
    "fasilitas": joblib.load('model_random_forest_fasilitas.pkl'),
    "pelayanan": joblib.load('model_random_forest_pelayanan.pkl'),
    "masakan": joblib.load('model_random_forest_masakan.pkl')
}

vectorizers = {
    "aspek": joblib.load('model_tfidf_aspek.pkl'),
    "fasilitas": joblib.load('tfidf_vectorizer_fasilitas.pkl'),
    "pelayanan": joblib.load('tfidf_vectorizer_pelayanan.pkl'),
    "masakan": joblib.load('tfidf_vectorizer_masakan.pkl')
}

# Load the 'finish_setelah_preprocessing.xlsx' file
df = pd.read_excel('finish_setelah_preprocessing.xlsx')

def main():
    st.title("Sistem Prediksi Aspek dan Sentimen dengan Random Forest")
    st.markdown("### Sistem ini memprediksi:\n- **Aspek**: Fasilitas, Pelayanan, Masakan\n- **Sentimen**: Positif atau Negatif")
    
    user_input = st.text_area("Masukkan Teks", "")
    
    if st.button("Prediksi"):
        processed_text = preprocess_text(user_input)
        aspect_vectorized = vectorizers["aspek"].transform([processed_text])
        predicted_aspect = aspect_model.predict(aspect_vectorized)[0]
        
        if predicted_aspect in sentiment_models:
            sentiment_vectorizer = vectorizers[predicted_aspect]
            sentiment_model = sentiment_models[predicted_aspect]
            sentiment_vectorized = sentiment_vectorizer.transform([processed_text])
            predicted_sentiment = sentiment_model.predict(sentiment_vectorized)[0]
            st.write(f"**Aspek**: {predicted_aspect.capitalize()}")
            st.write(f"**Sentimen**: {predicted_sentiment.capitalize()}")
        else:
            st.error("Aspek tidak dikenali.")

if __name__ == "__main__":
    main()
