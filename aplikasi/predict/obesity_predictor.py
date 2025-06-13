# Import joblib untuk memuat model dan encoder yang sudah disimpan
import joblib

# Kelas untuk memuat model yang sudah dilatih dan melakukan prediksi
class ObesityPredictor:
    def __init__(self, model_path, encoder_path):
        # Memuat pipeline model dari file
        self.model = joblib.load(model_path)
        # Memuat label encoder dari file
        self.label_encoder = joblib.load(encoder_path)

    # Method untuk melakukan prediksi pada data input X (DataFrame)
    def predict(self, X):
        # Data fitur yang dibutuhkan untuk prediksi
        required_features = ['Age', 'Gender', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']

        # Memastikan semua fitur yang dibutuhkan tersedia dalam input
        if not all(col in X.columns for col in required_features):
            raise ValueError(f"Missing required features. Required: {required_features}")

        # Melakukan prediksi kategori (dalam bentuk label numerik)
        prediction = self.model.predict(X)
        # Mengambil probabilitas dari setiap kelas prediksi
        probabilities = self.model.predict_proba(X)
        # Mengubah label numerik kembalu ke label asli menggunakan label encoder
        label = self.label_encoder.inverse_transform(prediction)

        # Mengembalikan label prediksi (nama kategori) dan probabilitasnya
        return label[0], probabilities[0]
