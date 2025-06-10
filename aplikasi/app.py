# Import library yang diperlukan dari Flask dan library lain
from flask import Flask, render_template, request
import joblib  # Untuk memuat model dan encoder
import pandas as pd  # Untuk manipulasi data
import os  # Untuk mengelola path file dan direktori

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Tentukan path ke direktori saat ini
BASE_DIR = os.path.dirname(__file__)

# Path ke model dan label encoder yang telah disimpan
MODEL_PATH = os.path.join(BASE_DIR, "model", "Saved_model", "obesity_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "Saved_model", "obesity_label_encoder.pkl")

# Memuat model machine learning dan label encoder dari file
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Route utama untuk halaman beranda (form input)
@app.route('/')
def home():
    return render_template('index.html')  # Menampilkan halaman index.html

# Route untuk memproses prediksi saat form disubmit
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil nilai input dari form pengguna dan mengonversinya ke tipe data yang sesuai
        age = float(request.form['age'])  # Usia
        gender = request.form['gender']  # Jenis kelamin
        height = float(request.form['height'])  # Tinggi badan
        weight = float(request.form['weight'])  # Berat badan
        bmi = float(request.form['bmi'])  # BMI (Body Mass Index)
        physical_activity = int(request.form['physical_activity'])  # Tingkat aktivitas fisik

        # Membuat dataframe dari input untuk diproses model
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Height': height,
            'Weight': weight,
            'BMI': bmi,
            'PhysicalActivityLevel': physical_activity
        }])

        # Melakukan prediksi dengan model (hasilnya berupa label numerik)
        prediction_encoded = model.predict(input_data)

        # Mengubah label numerik ke nama kategori aslinya
        prediction_label = label_encoder.inverse_transform(prediction_encoded)

        # Menampilkan hasil prediksi ke halaman result.html
        return render_template('result.html', prediction=prediction_label[0])
    
    # Jika terjadi error saat proses prediksi, tampilkan pesan error ke pengguna
    except Exception as e:
        return render_template('result.html', prediction=f"Terjadi error: {str(e)}")

# Menjalankan aplikasi Flask dalam mode debug
if __name__ == '__main__':
    app.run(debug=True)
