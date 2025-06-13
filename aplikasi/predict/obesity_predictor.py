# Import joblib untuk memuat model dan encoder yang sudah disimpan
import joblib

# Kelas untuk memuat model yang sudah dilatih dan melakukan prediksi
class ObesityPredictor:
    def __init__(self, model_path, encoder_path):
        # Memuat pipeline model dari file
        self.model = joblib.load(model_path)
        # Memuat label encoder dari file
        self.label_encoder = joblib.load(encoder_path)

    # Method untuk melakukan prediksi pada data 
    def predict(self, X):
        required_features = ['Age', 'Gender', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']
        if not all(col in X.columns for col in required_features):
            raise ValueError(f"Missing required features. Required: {required_features}")

        prediction = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        label = self.label_encoder.inverse_transform(prediction)
        return label[0], probabilities[0]
