# TUGAS-BESAR-PERANCANGAN-APLIKASI-SAINS-DATA

<PRE>
   Nama Anggota Kelompok:
   1. Ika Wida Nuragustin (2311110001)
   2. Chelsisdeo A.P Sanenek (2311110016)
   3. Indy Aurellia Az Zhara (2311110020)
   4. Dill Thafa Jausha (2311110048)
</PRE>

---

# Deskripsi Aplikasi

Website ini adalah aplikasi berbasis web yang bertujuan untuk 
memprediksi kategori obesitas seseorang berdasarkan data pribadi dan gaya 
hidup. Prediksi berdasarkan beberapa fitur, yaitu Age, Gender, Height, 
Weight, BMI, dan Physical Activity Level. Model machine learning yang 
digunakan adalah Random Forest, dilatih dengan preprocessing 
menggunakan Pipeline Scikit-learn.  

*Fitur Utama:

- Formulir Input Pengguna

Pengguna mengisi data seperti umur, tinggi, berat, jenis kelamin, BMI, dan tingkat aktivitas fisik melalui halaman web.

- Prediksi Langsung

Setelah pengguna menekan tombol Submit, data dikirim ke server Flask, diproses oleh model machine learning, dan hasil prediksi kategori obesitas ditampilkan secara langsung.

- Model Pipeline
  
Sistem menggunakan pipeline yang mencakup:

-> Pra-pemrosesan data (scaling, encoding)

-> Klasifikasi dengan Logistic Regression

-> Label encoder untuk mengembalikan prediksi ke label aslinya

-> Struktur Modular

-> obesity_model_handler.py: untuk training dan penyimpanan model

-> obesity_predictor.py: untuk prediksi data baru

-> app.py: frontend dan routing Flask

-> main.py: skrip utama untuk training dan demo prediksi

---

## Pelatihan Model

### 1. Prasyarat
# Pastikan Anda telah menginstal:
- Python 3.8+
- Library Python yang diperlukan: `pandas`, `scikit-learn`, `joblib`, `numpy`

Instal dependensi dengan perintah:
```bash
pip install pandas scikit-learn joblib numpy
```

### 2. Dataset
Tempatkan file dataset `obesity_data.csv` di direktori root proyek. Pastikan dataset berisi kolom-kolom berikut:
- `Age`
- `Gender`
- `Height`
- `Weight`
- `BMI`
- `PhysicalActivityLevel`
- `ObesityCategory` (variabel target)

### 3. Melatih Model
Jalankan skrip `obesity_model_handler.py` untuk melatih model:
```bash
python obesity_model_handler.py
```

Skrip ini melakukan langkah-langkah berikut:
1. **Memuat dataset**: Membaca file `obesity_data.csv` ke dalam memori untuk diproses.
2. **Pra-pemrosesan data**: Termasuk penskalaan fitur numerik (misalnya tinggi, berat) dan encoding fitur kategorikal (misalnya gender). Pra-pemrosesan memastikan data dalam format yang cocok untuk algoritma machine learning.
3. **Membagi data**: Membagi dataset menjadi data pelatihan dan pengujian untuk mengevaluasi performa model terhadap data yang belum pernah dilihat.
4. **Melatih model regresi logistik**: Melatih model regresi logistik pada data pelatihan untuk memprediksi kategori obesitas.
5. **Evaluasi model**: Menguji model pada data pengujian untuk mengukur akurasi dan kinerjanya.
6. **Menyimpan model dan artefak pra-pemrosesan**: File berikut disimpan untuk kebutuhan deployment:
   - `obesity_model.pkl`: Model regresi logistik yang telah dilatih. Digunakan untuk prediksi saat deployment.
   - `obesity_preprocessor.pkl`: Pipeline pra-pemrosesan yang menjamin data input saat deployment diproses sama seperti saat pelatihan.

### 4. Output
Setelah menjalankan skrip, file-file berikut akan dihasilkan:
- `obesity_model.pkl`: Model yang telah dilatih.
- `obesity_preprocessor.pkl`: Pipeline pra-pemrosesan.
- 
File-file ini penting untuk deployment karena menjamin konsistensi antara fase pelatihan dan prediksi.

#### Contoh Penggunaan:
- **`obesity_model.pkl`**:
  File ini digunakan untuk melakukan prediksi. Contoh:
  ```python
  import joblib
  model = joblib.load('obesity_model.pkl')
  prediction = model.predict([[25, 1.75, 70, 2]])  # Contoh input: Usia, Tinggi, Berat, Tingkat Aktivitas Fisik
  print(prediction)  # Output berupa label numerik kategori obesitas
  ```

- **`obesity_preprocessor.pkl`**:
  File ini menjamin data input diproses seperti saat pelatihan. Contoh:
  ```python
  preprocessor = joblib.load('obesity_preprocessor.pkl')
  processed_data = preprocessor.transform([[25, 'Male', 1.75, 70, 2]])  # Contoh input: Usia, Gender, Tinggi, Berat, Aktivitas Fisik
  ```

- **`obesity_target_encoder.pkl`**:
  File ini mengubah prediksi numerik kembali ke label kategori aslinya. Contoh:
  ```python
  target_encoder = joblib.load('obesity_target_encoder.pkl')
  category = target_encoder.inverse_transform([2])  # Contoh input: Label numerik
  print(category)  # Output kategori yang sesuai, misalnya "Overweight"
  ```

---

## Deployment Model

### 1. Prasyarat
Pastikan Anda telah menginstal:
- Flask

Instal Flask dengan perintah:
```bash
pip install flask
```

### 2. Pengaturan Aplikasi
Deployment dilakukan melalui skrip `app.py`. Skrip ini menggunakan Flask untuk membuat aplikasi web yang memprediksi kategori obesitas.

### 3. Menjalankan Aplikasi
Mulai aplikasi Flask dengan menjalankan:
```bash
python app.py
```

Aplikasi akan dapat diakses melalui `http://127.0.0.1:5000/`.

### 4. Fitur Aplikasi
- **Halaman Utama**: Formulir untuk menginput data pengguna (misalnya: Usia, Gender, Tinggi, Berat, BMI, Tingkat Aktivitas Fisik).
- **Endpoint Prediksi**: Mengirim data formulir ke endpoint `/predict`, yang akan mengembalikan kategori obesitas yang diprediksi beserta probabilitasnya.

---

## Diagram
Berikut adalah diagram tingkat tinggi dari pipeline:

```plaintext
+-------------------+       +-------------------+       +-------------------+
|   Pemuatan Data   | --->  |   Pra-pemrosesan  | --->  |   Pelatihan Model |
+-------------------+       +-------------------+       +-------------------+

+-------------------+       +-------------------+
|  Backend Flask    | <-->  |  Model Terlatih   |
+-------------------+       +-------------------+
```

---

## Catatan
- Pastikan dataset bersih dan bebas dari nilai yang hilang sebelum melatih model.
- Modifikasi skrip `app.py` sesuai kebutuhan untuk menyesuaikan aplikasi web.

---

Jika ada pertanyaan atau masalah, silakan hubungi pengelola proyek.
EOF
