import joblib

class ObesityPredictor:
    def __init__(self, model_path, encoder_path):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)

    def predict(self, X):
        required_features = ['Age', 'Gender', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']
        if not all(col in X.columns for col in required_features):
            raise ValueError(f"Missing required features. Required: {required_features}")

        prediction = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        label = self.label_encoder.inverse_transform(prediction)
        return label[0], probabilities[0]
