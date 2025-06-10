# Obesity Prediction Pipeline

README ini menyediakan panduan langkah demi langkah untuk membangun Obesity Prediction Pipeline, termasuk pelatihan model dan deployment. Ikuti instruksi di bawah ini untuk menyiapkan proyek.

---

## Model Training

### 1. Prerequisites
# Ensure you have the following installed:
- Python 3.8+
- Required Python libraries: `pandas`, `scikit-learn`, `joblib`, `numpy`

Install the dependencies using:
```bash
pip install pandas scikit-learn joblib numpy
```

### 2. Dataset
Place the dataset file `obesity_data.csv` in the project root directory. Ensure the dataset contains the following columns:
- `Age`
- `Gender`
- `Height`
- `Weight`
- `BMI`
- `PhysicalActivityLevel`
- `ObesityCategory` (target variable)

### 3. Training the Model
Run the `obesity_model_pipeline.py` script to train the model:
```bash
python obesity_model_pipeline.py
```

This script performs the following steps:
1. **Loads the dataset**: Reads the `obesity_data.csv` file into memory for processing.
2. **Preprocesses the data**: This step includes scaling numerical features (e.g., height, weight) and encoding categorical features (e.g., gender). Preprocessing ensures that the data is in a format suitable for machine learning algorithms.
3. **Splits the data**: Divides the dataset into training and testing sets to evaluate the model's performance on unseen data.
4. **Trains a logistic regression model**: Fits a logistic regression model to the training data to predict obesity categories.
5. **Evaluates the model**: Tests the model on the test set to measure its accuracy and performance.
6. **Saves the trained model and preprocessing artifacts**: The following files are saved for deployment:
   - `obesity_model.pkl`: This file contains the trained logistic regression model. It is used to make predictions during deployment.
   - `obesity_preprocessor.pkl`: This file contains the preprocessing pipeline, which ensures that input data during deployment is transformed in the same way as during training.
   - `obesity_target_encoder.pkl`: This file contains the label encoder for the target variable (`ObesityCategory`). It maps the predicted numerical labels back to their original categorical values (e.g., "Normal", "Overweight").

### 4. Outputs
After running the script, the following files will be generated:
- `obesity_model.pkl`: Trained model.
- `obesity_preprocessor.pkl`: Preprocessing pipeline.
- `obesity_target_encoder.pkl`: Label encoder for the target variable.

These files are essential for deployment as they ensure consistency between training and prediction phases.

#### Examples of Usage:
- **`obesity_model.pkl`**:
  This file is used to make predictions. For example:
  ```python
  import joblib
  model = joblib.load('obesity_model.pkl')
  prediction = model.predict([[25, 1.75, 70, 2]])  # Example input: Age, Height, Weight, Physical Activity Level
  print(prediction)  # Outputs the predicted obesity category as a numerical label
  ```

- **`obesity_preprocessor.pkl`**:
  This file ensures that input data is preprocessed in the same way as during training. For example:
  ```python
  preprocessor = joblib.load('obesity_preprocessor.pkl')
  processed_data = preprocessor.transform([[25, 'Male', 1.75, 70, 2]])  # Example input: Age, Gender, Height, Weight, Physical Activity Level
  ```

- **`obesity_target_encoder.pkl`**:
  This file maps numerical predictions back to their original categorical labels. For example:
  ```python
  target_encoder = joblib.load('obesity_target_encoder.pkl')
  category = target_encoder.inverse_transform([2])  # Example input: Numerical label
  print(category)  # Outputs the corresponding category, e.g., "Overweight"
  ```

---

## Model Deployment

### 1. Prerequisites
Ensure you have the following installed:
- Flask

Install Flask using:
```bash
pip install flask
```

### 2. Application Setup
The deployment is handled by the `app.py` script. This script uses Flask to create a web application for predicting obesity categories.

### 3. Running the Application
Start the Flask application by running:
```bash
python app.py
```

The application will be accessible at `http://127.0.0.1:5000/`.

### 4. Application Features
- **Home Page**: A form to input user data (e.g., Age, Gender, Height, Weight, BMI, Physical Activity Level).
- **Prediction Endpoint**: Submits the form data to the `/predict` endpoint, which returns the predicted obesity category and probabilities.

---

## Diagram
Below is a high-level diagram of the pipeline:

```plaintext
+-------------------+       +-------------------+       +-------------------+
|   Data Loading    | --->  |   Preprocessing   | --->  |   Model Training  |
+-------------------+       +-------------------+       +-------------------+

+-------------------+       +-------------------+
|   Flask Backend   | <-->  |   Trained Model   |
+-------------------+       +-------------------+
```

---

## Notes
- Ensure the dataset is clean and free of missing values before training.
- Modify the `app.py` script to customize the web application as needed.

---

For any issues or questions, feel free to contact the project maintainer.
