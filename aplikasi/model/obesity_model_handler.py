import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

class ObesityModelHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = None
        self.label_encoder = None
        self.X = None
        self.y = None

    def load_data(self):
        data = pd.read_csv(self.file_path)
        self.X = data.drop("ObesityCategory", axis=1)
        self.y = data["ObesityCategory"]
        return data

    def preprocess(self):
        categorical_features = ['Gender']
        numerical_features = ['Age', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']

        preprocessor = ColumnTransformer([
            ('numerical', Pipeline([('scaler', StandardScaler())]), numerical_features),
            ('categorical', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])

        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
        ])

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

    def save_all(self, model_path, encoder_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
