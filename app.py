import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Fungsi Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Memuat Model Random Forest
aspect_model = joblib.load('aspek.pkl')

sentiment_models = {
    "fasilitas": joblib.load('model_random_forest_fasilitas.pkl'),
    "pelayanan": joblib.load('model_random_forest_pelayanan.pkl'),
    "masakan": joblib.load('model_random_forest_masakan.pkl')
}

# Memuat TF-IDF Vectorizer
vectorizers = {
    "aspek": joblib.load('tfidfaspek.pkl'),
    "fasilitas": joblib.load('tfidf_vectorizer_fasilitas.pkl'),
    "pelayanan": joblib.load('tfidf_vectorizer_pelayanan.pkl'),
    "masakan": joblib.load('tfidf_vectorizer_masakan.pkl')
}

# Aplikasi Streamlit
def main():
    st.title("Sistem Prediksi Aspek dan Sentimen dengan Random Forest")
    st.markdown("### Sistem ini memprediksi:\n- **Aspek**: Fasilitas, Pelayanan, Masakan\n- **Sentimen**: Positif atau Negatif")
    
    # Input dari pengguna
    user_input = st.text_area("Masukkan Teks", "")
    
    # Tombol Prediksi
    if st.button("Prediksi"):
        # Preprocessing
        processed_text = preprocess_text(user_input)
        
        # Prediksi Aspek
        aspect_vectorized = vectorizers["aspek"].transform([processed_text])
        predicted_aspect = aspect_model.predict(aspect_vectorized)[0]
        
        # Prediksi Sentimen
        sentiment_vectorizer = vectorizers[predicted_aspect]
        sentiment_model = sentiment_models[predicted_aspect]
        sentiment_vectorized = sentiment_vectorizer.transform([processed_text])
        predicted_sentiment = sentiment_model.predict(sentiment_vectorized)[0]
        
        # Menampilkan hasil prediksi
        st.write(f"**Aspek**: {predicted_aspect.capitalize()}")
        st.write(f"**Sentimen**: {predicted_sentiment.capitalize()}")

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
