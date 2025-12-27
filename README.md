# 📊 Clustering Status Gizi Penduduk di Indonesia  
### Berdasarkan Tingkat Kemiskinan Menggunakan K-Means, K-Median, dan CLARA

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk melakukan **pengelompokan (clustering)** wilayah di Indonesia berdasarkan **status gizi penduduk dan tingkat kemiskinan**. Analisis ini diharapkan dapat membantu dalam memahami pola ketimpangan gizi serta menjadi dasar pengambilan kebijakan yang lebih tepat sasaran.

Metode clustering yang digunakan adalah:
- **K-Means**
- **K-Median**
- **CLARA (Clustering Large Applications)**

Pendekatan ini relevan dengan **Sustainable Development Goals (SDGs)**, khususnya:
- **SDG 2: Zero Hunger**
- **SDG 3: Good Health and Well-Being**

---

## 📂 Variabel yang Digunakan
Dataset yang digunakan mencakup indikator berikut:

| Variabel | Deskripsi |
|--------|----------|
| PoU | Prevalensi penduduk yang kekurangan konsumsi/gizi |
| Jumlah Penduduk | Total jumlah penduduk per wilayah |
| Penduduk Undernourish | Jumlah penduduk dengan kondisi kurang gizi |
| Persentase Penduduk Miskin (P0) | Persentase penduduk di bawah garis kemiskinan |

---

## 🧠 Metodologi
Tahapan analisis dalam proyek ini meliputi:

1. **Data Collection**  
   Menggunakan data statistik terkait gizi dan kemiskinan penduduk di Indonesia.

2. **Data Preprocessing**
   - Handling missing values
   - Normalisasi data (Min-Max Scaling)

3. **Clustering**
   - Implementasi **K-Means**
   - Implementasi **K-Median**
   - Implementasi **CLARA**

4. **Evaluasi Cluster**
   - Silhouette Coefficient
   - Davies-Bouldin Index

5. **Interpretasi Hasil**
   - Analisis karakteristik setiap cluster
   - Identifikasi wilayah dengan risiko gizi tinggi

---

## 📈 Output yang Dihasilkan
- Label cluster untuk setiap wilayah
- Perbandingan performa K-Means, K-Median, dan CLARA
- Visualisasi hasil clustering
- Insight terkait hubungan kemiskinan dan status gizi penduduk
- pemetaan wilayah di indonesia berdasarkan indikator status gizi

---

## 🎯 Manfaat Perancangan
- memberikan informasi ke publik mengenai status gizi indonesia
- membantu peneliti sebagai refrensi pengembangan kedepanya

---
## 🛠️ Requirements & Installation

Untuk menjalankan aplikasi ini, pastikan Anda telah menginstal **Python 3.8+** dan library berikut.

### 📦 Library yang Digunakan
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- folium
- pillow
- openpyxl

---

### 📥 Cara Instalasi

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/alfianindra/Clustering-Wilayah.git
```

#### Download library yang diperlukan pada command prompt 
``` bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn folium pillow openpyxl
```

## 👨‍🎓 Author
**Alfian Indrajaya**  
Information Technology Student
Interest: Data Analysis & Web Development  

---

## 📄 Lisensi
Proyek ini dibuat untuk keperluan akademik dan penelitian.
