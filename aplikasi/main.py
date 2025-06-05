from model.obesity_model_handler import ObesityModelHandler
from predict.obesity_predictor import ObesityPredictor
import pandas as pd
import os

# Pastikan path selalu benar meskipun script dijalankan dari mana saja
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "Saved_model", "obesity_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "Saved_model", "obesity_label_encoder.pkl")
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "obesity_data.csv"))

# Buat folder Saved_model kalau belum ada
os.makedirs(os.path.join(BASE_DIR, "model", "Saved_model"), exist_ok=True)

# Training dan simpan model
handler = ObesityModelHandler(DATA_PATH)
handler.load_data()
handler.preprocess()
handler.train()
handler.save_all(MODEL_PATH, ENCODER_PATH)

# Prediksi contoh baru
predictor = ObesityPredictor(MODEL_PATH, ENCODER_PATH)

new_sample = pd.DataFrame([{
    'Age': 30,
    'Gender': 'Male',
    'Height': 175,
    'Weight': 70,
    'BMI': 22.86,
    'PhysicalActivityLevel': 3
}])

label, proba = predictor.predict(new_sample)
print(f"\nPredicted Obesity Category: {label}")
print("Prediction Probabilities:", proba)
