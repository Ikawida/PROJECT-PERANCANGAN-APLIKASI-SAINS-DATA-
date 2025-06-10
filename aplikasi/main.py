# Import class handler untuk proses training dan penyimpanan model
from model.obesity_model_handler import ObesityModelHandler

# Import class predictor untuk melakukan prediksi menggunakan model yang sudah dilatih
from predict.obesity_predictor import ObesityPredictor

# Import pandas untuk manipulasi data
import pandas as pd

# Import os untuk menangani path dan struktur folder
import os

# Menentukan direktori dasar (path dari file ini) agar tetap konsisten saat dijalankan dari mana pun
BASE_DIR = os.path.dirname(__file__)

# Menentukan path untuk model yang akan disimpan
MODEL_PATH = os.path.join(BASE_DIR, "model", "Saved_model", "obesity_model.pkl")

# Menentukan path untuk label encoder yang akan disimpan
ENCODER_PATH = os.path.join(BASE_DIR, "model", "Saved_model", "obesity_label_encoder.pkl")

# Menentukan path absolut ke dataset untuk training
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "obesity_data.csv"))

# Membuat folder 'Saved_model' jika belum ada, untuk menyimpan model dan encoder
os.makedirs(os.path.join(BASE_DIR, "model", "Saved_model"), exist_ok=True)

# Inisialisasi handler model dengan path ke dataset
handler = ObesityModelHandler(DATA_PATH)

# Memuat data dari file CSV
handler.load_data()

# Melakukan preprocessing terhadap data (encoding, scaling, dll)
handler.preprocess()

# Melatih model menggunakan data yang telah diproses
handler.train()

# Menyimpan model dan label encoder ke file
handler.save_all(MODEL_PATH, ENCODER_PATH)

# Inisialisasi predictor menggunakan model dan encoder yang telah disimpan
predictor = ObesityPredictor(MODEL_PATH, ENCODER_PATH)

# Contoh data baru untuk diprediksi (berupa DataFrame)
new_sample = pd.DataFrame([{
    'Age': 30,  # Usia
    'Gender': 'Male',  # Jenis kelamin
    'Height': 175,  # Tinggi badan dalam cm
    'Weight': 70,  # Berat badan dalam kg
    'BMI': 22.86,  # Body Mass Index
    'PhysicalActivityLevel': 3  # Tingkat aktivitas fisik (skala)
}])

# Melakukan prediksi terhadap contoh data baru
label, proba = predictor.predict(new_sample)

# Menampilkan hasil prediksi kategori obesitas
print(f"\nPredicted Obesity Category: {label}")

# Menampilkan probabilitas dari setiap kategori prediksi
print("Prediction Probabilities:", proba)
