# Early Warning System for Student Dropout Risk

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.


### Permasalahan Bisnis

11. **Tingginya tingkat dropout mahasiswa** yang dapat memengaruhi reputasi dan kualitas pendidikan di Jaya Jaya Institut.

2. **Tidak adanya sistem deteksi dini** untuk mengidentifikasi mahasiswa yang berisiko drop out.

3. **Sulitnya menentukan faktor-faktor yang mempengaruhi dropout**

4. **Tidak ada sistem pemantauan perkembangan mahasiswa** yang komprehensif untuk mendukung keputusan intervensi.

5. **Kurangnya intervensi atau bimbingan tepat waktu** bagi mahasiswa yang berisiko agar tetap melanjutkan pendidikan mereka.


### Cakupan Proyek

Cakupan proyek ini mencakup seluruh proses analisis data hingga evaluasi model prediksiMahasiswa Dropout pada Jaya Institute. Adapun cakupan kegiatan dalam proyek ini meliputi:

### 1. Persiapan
- Menyiapkan library yang dibutuhkan.
- Menyiapkan dan memuat dataset yang akan digunakan.

### 2. Data Understanding
- **Exploratory Data Analysis (EDA)**:
  - *Univariate Analysis*: Analisis distribusi masing-masing fitur.
  - *Bivariate Analysis*: Analisis hubungan antar fitur dan target `Status`.

### 3. Data Preparation / Preprocessing
- Menghapus outlier.
- Seleksi fitur yang relevan.
- Encoding variabel kategorikal.
- Pembagian data menjadi *train* dan *test set*.
- Standarisasi fitur numerik.
- Penerapan PCA untuk reduksi data


### 4. Modeling
Pembangunan model klasifikasi untuk memprediksi `Status` menggunakan beberapa algoritma:
- **Random Forest**
- **Naive Bayes (NB)**
- **XGBoost (XGB)**

### 5. Evaluation
- Evaluasi model dilakukan sebelum dan sesudah tuning **Hyperparameter**.
- Perbandingan performa antar model untuk memilih model terbaik.


Proyek ini akan menghasilkan model prediksi yang dapat membantu institusi dalam mengidentifikasi mahasiswa berisiko tinggi untuk terkena Dropout, serta menyediakan dasar pengambilan keputusan yang lebih tepat.

### Persiapan

Sumber data: [Click Here!](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

# Setup Environment

Lingkungan pengembangan dan analisis proyek ini menggunakan Python serta tool visualisasi Metabase. Berikut langkah-langkah setup yang perlu dilakukan:

---

## **1. Membuat dan Mengaktifkan Virtual Environment (venv)**

Sebelum menginstal dependensi, pastikan Anda membuat dan mengaktifkan virtual environment untuk proyek ini. 

### Langkah-langkah:

1. **Buat Virtual Environment**  
   Masuk ke direktori proyek Anda menggunakan terminal, lalu buat virtual environment dengan perintah berikut:

   python -m venv venv

Anda bisa mengganti nama `venv` dengan nama lain jika diinginkan.

2. **Aktifkan Virtual Environment**

- **Untuk Windows**, jalankan perintah:
  ```
  .\venv\Scripts\activate
  ```

- **Untuk macOS/Linux**, jalankan perintah:
  ```
  source venv/bin/activate
  ```

3. Setelah virtual environment aktif, Anda akan melihat nama `venv` muncul di awal prompt terminal Anda.

---

## **2. Install Library Python**

Untuk memastikan semua dependensi Python terinstal, gunakan file `requirements.txt` yang telah disediakan. Jalankan perintah berikut di terminal:

pip install -r requirements.txt


---

## **3. Menjalankan Business Dashboard dengan Metabase**

Metabase digunakan untuk membuat dashboard interaktif yang menampilkan insight dari data karyawan.

### 🔧 Langkah Setup Metabase:

1. **Tarik Image Metabase menggunakan Docker:**

   docker pull metabase/metabase:v0.46.4


2. **Jalankan Container Metabase:**

   docker run -p 3002:3000 --name submisson_2 metabase/metabase


3. **Akses Metabase melalui Browser:**

   http://localhost:3002


4. **Login ke Metabase:**

   Username: root@mail.com
   Password: root123
---

## Business Dashboard
Jelaskan tentang business dashboard yang telah dibuat. Jika ada, sertakan juga link untuk mengakses dashboard tersebut.

## Menjalankan Sistem Machine Learning
Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.

## **4. Pengujian Model**

Pengujian model dapat dilakukan pada link berikut : [Click Here!](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)


```

```

## Conclusion
Jelaskan konklusi dari proyek yang dikerjakan.

### Rekomendasi Action Items
Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.
- action item 1
- action item 2
