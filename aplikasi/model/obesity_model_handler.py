# Import library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split #Untuk memmbagi data latih dan data uji
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder # Unuk preprocessing data
from sklearn.compose import ColumnTransformer # Untuk menggabungkan preprocessing numerik dan kategorik
from sklearn.linear_model import LogisticRegression # Model klasifikasi
from sklearn.pipeline import Pipeline # Untuk membuat alur pemrosesan data dan model
from sklearn.metrics import accuracy_score, classification_report # UNtuk evaluasi model
import joblib # Untuk menyimpan model dan encoder ke file

# Kelas untuk menangani pemrosesan dan pelatihan model obesitas
class ObesityModelHandler:
    def __init__(self, file_path):
        self.file_path = file_path # Path ke file CSV yang berisi data
        self.model = None # Tempat menyimpan model pipeline
        self.label_encoder = None # Untuk menyimpan encoder target
        self.X = None # Variabel fitur
        self.y = None # Variabel target

    # Method untuk memuat data dari CSV dan memisahkan fitur dan label
    def load_data(self):
        data = pd.read_csv(self.file_path) # Membaca file CSV
        self.X = data.drop("ObesityCategory", axis=1) # Menghapus kolom target dari fitur
        self.y = data["ObesityCategory"] # Menyimpan kolom target
        return data # Mengembalikan DataFrame

    # Method untuk menyiapkan preprocessing dan pipeline model
    def preprocess(self):
        # Menentukan fitur kategorik dan numerik
        categorical_features = ['Gender']
        numerical_features = ['Age', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']

        #M Menyusun preprocessing untuk numerik dan kategorik
        preprocessor = ColumnTransformer([
            # Fitur numerik akan diskalakan menggunakan StandardScaler
            ('numerical', Pipeline([('scaler', StandardScaler())]), numerical_features),
            # Fitur kategorik diubah menjadi one-hot encoding
            ('categorical', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])

        # Label target dikodekan ke angka menggunakan LabelEncoder
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

        #M Membuat pipeline yang menggabungkan preprocessing dan model klasifikasi
        self.model = Pipeline([
            ('preprocessor', preprocessor), # Langkah preprocesiing
            ('classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)) # Model regresi logistik
        ])

    # Method untuk melatih model dan menampilakn evaluasinya
    def train(self):
        # Membagi data menjadi data latih dan data uji dengan proporsi 80:20 dan stratifikasi label
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        # Melatih model pipeline
        self.model.fit(X_train, y_train)

        # Memprediksi data uji
        y_pred = self.model.predict(X_test)

        # Menampilkan akurasi dan laporan klasifikasi berdasarkan 
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

    # Method untuk menyimpan model pipeline dan label encoder ke file
    def save_all(self, model_path, encoder_path):
        joblib.dump(self.model, model_path) # Menyimpan pipeline model
        joblib.dump(self.label_encoder, encoder_path) # Menyimpan label encoder
