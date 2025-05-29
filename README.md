# Diagnosis Diabetes dengan Decision Tree

Repo ini berisi script Python untuk melakukan analisis dan prediksi diabetes menggunakan model Decision Tree.  
Data yang digunakan adalah dataset `diabetes.csv` yang berisi informasi pasien seperti kadar glukosa, tekanan darah, BMI, dll.

## 📂 Isi Repo
- `diabetestdt.py` → Script utama untuk:
    - Memuat dataset
    - Membersihkan data (data preprocessing)
    - Melatih model Decision Tree
    - Mengevaluasi akurasi model
    - Membuat visualisasi pohon keputusan dan menyimpannya ke file PNG
- `diabetes.csv` → Dataset input (pastikan file ini ada di direktori yang sama)
- `.github/workflows/run-diabetes.yml` → File workflow GitHub Actions untuk menjalankan script secara otomatis saat ada push

## 🚀 Cara Menjalankan Script
1. Pastikan sudah install library yang dibutuhkan:
    ```
    pip install pandas matplotlib scikit-learn
    ```

2. Jalankan script:
    ```
    python diabetestdt.py
    ```

3. Hasil evaluasi model akan muncul di terminal, dan visualisasi pohon keputusan akan disimpan sebagai:
    ```
    decision_tree.png
    ```

## 🛠️ Workflow GitHub Actions
Setiap kali ada push ke repo, workflow otomatis akan:
- Menjalankan `diabetestdt.py`
- Menyimpan visualisasi `decision_tree.png` sebagai **artifact** di tab Actions

## 📊 Output
- **Akurasi Model**
- **Confusion Matrix**
- **Classification Report**
- **Visualisasi Pohon Keputusan** (bisa di-download dari artifact GitHub Actions)

## ❤️ Credits
Project ini dibuat oleh [mutimrd2n](https://github.com/mutimrd2n)  
Menggunakan dataset publik Pima Indians Diabetes Dataset.
