import streamlit as st
from pathlib import Path
from style import load_custom_style

# === Konfigurasi Halaman ===
st.set_page_config(
    page_title="Clustering Wilayah di Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Load CSS Global ===
load_custom_style()

# === Judul Halaman ===
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0;'>🌏 Clustering Wilayah di Indonesia</h1>
    <h3 style='text-align: center; color: #555;'>Berdasarkan Status Gizi Penduduk</h3>
    """,
    unsafe_allow_html=True
)

# === Tujuan Website ===
st.markdown(
    """
    <div style="display: flex; justify-content: center; margin-top: 30px; margin-bottom: 40px;">
        <div class="card" style="max-width: 850px; text-align: justify; padding: 25px 30px;">
            <div class="card-title" style="text-align: center;">🎯 Tujuan Website</div>
            <div>
                Website ini bertujuan untuk <b>memvisualisasikan hasil pengelompokan (clustering) wilayah di Indonesia</b> 
                berdasarkan indikator <b>status gizi penduduk</b>. 
                Dengan pendekatan ini, pengguna dapat memahami pola dan perbedaan tingkat gizi antar wilayah, 
                serta mengidentifikasi daerah yang membutuhkan perhatian khusus dalam upaya peningkatan kesejahteraan gizi masyarakat. <br><br>
                Selain itu, website ini berfungsi sebagai <b>alat bantu analisis</b> bagi peneliti, pembuat kebijakan, 
                dan masyarakat umum dalam melihat kondisi gizi nasional secara <b>interaktif, informatif, dan visual</b>.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# === Layout 2 atas dan 1 bawah ===
st.markdown("<hr style='margin-top:10px;margin-bottom:30px;'>", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
        <div class="card">
            <div class="card-title">🍎 Pentingnya Gizi</div>
            <div>
                Gizi merupakan aspek fundamental dalam menentukan kualitas hidup masyarakat. 
                Status gizi yang baik berkontribusi terhadap <b>kesehatan, produktivitas, dan kesejahteraan</b>. 
                Melalui analisis data gizi, kita dapat memahami pola ketimpangan dan potensi perbaikan di berbagai wilayah.
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="card">
            <div class="card-title">🧩 Metode Clustering</div>
            <div>
                Website ini menggunakan tiga metode utama dalam proses pengelompokan data:
                    <b>K-Means</b>
                    <b>K-Median</b>
                    <b>CLARA (Clustering Large Applications)</b></li>
                Masing-masing metode memiliki keunggulan tersendiri dalam mengolah dan mengelompokkan data gizi wilayah.
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin-top:40px;margin-bottom:30px;'>", unsafe_allow_html=True)
col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown("""
        <div class="card">
            <div class="card-title">📊 K-Means</div>
            <div>
                K-Means membagi data ke dalam beberapa kelompok (cluster) berdasarkan kemiripan nilai fitur. 
                Tujuan utamanya adalah <b>meminimalkan jarak antar data dalam cluster</b> dan mencari pusat yang paling representatif.
            </div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="card">
            <div class="card-title">📉 K-Median</div>
            <div>
                K-Median menggunakan <b>median</b> sebagai pusat cluster. 
                Pendekatan ini membuatnya lebih tahan terhadap <b>outlier</b> dan cocok untuk data dengan distribusi tidak merata.
            </div>
        </div>
    """, unsafe_allow_html=True)

# === CLARA Section + Buku Panduan Button ===
st.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center; margin-top: 40px;">
        <div class="card" style="width: 60%; margin-bottom: 20px;">
            <div class="card-title">⚙️ CLARA (Clustering Large Applications)</div>
            <div>
                CLARA dirancang untuk <b>menangani dataset besar</b> dengan efisien. 
                Algoritma ini bekerja dengan mengambil sampel acak dari data, menerapkan metode K-Medoids pada sampel, 
                dan mengulangi proses tersebut untuk menemukan hasil clustering terbaik dari keseluruhan data.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)