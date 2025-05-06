from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model, preprocessor, and label encoder
model = joblib.load("obesity_model.pkl")
preprocessor = joblib.load("obesity_preprocessor.pkl")
label_encoder = joblib.load("obesity_target_encoder.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = float(request.form['age'])
    gender = request.form['gender']
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    bmi = float(request.form['bmi'])
    physical_activity = int(request.form['physical_activity'])

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'BMI': [bmi],
        'PhysicalActivityLevel': [physical_activity]
    })

    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)

    # Make a prediction
    prediction_encoded = model.predict(processed_data)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    # Return the result
    return render_template('result.html', prediction=prediction_label[0])

if __name__ == '__main__':
    app.run(debug=True)