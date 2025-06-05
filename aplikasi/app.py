from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model dan label encoder dari folder 'model'
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "Saved_model", "obesity_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "Saved_model", "obesity_label_encoder.pkl")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        age = float(request.form['age'])
        gender = request.form['gender']
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        bmi = float(request.form['bmi'])
        physical_activity = int(request.form['physical_activity'])

        # Buat dataframe untuk prediksi
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Height': height,
            'Weight': weight,
            'BMI': bmi,
            'PhysicalActivityLevel': physical_activity
        }])

        # Prediksi
        prediction_encoded = model.predict(input_data)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)

        return render_template('result.html', prediction=prediction_label[0])
    except Exception as e:
        return render_template('result.html', prediction=f"Terjadi error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
