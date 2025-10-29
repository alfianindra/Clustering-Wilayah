import streamlit as st
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
    <h2 style='text-align: center;'>Clustering Wilayah di Indonesia</h2>
    <h3 style='text-align: center;'>Berdasarkan Status Gizi Penduduk</h3>
    """,
    unsafe_allow_html=True
)

# === Tujuan Website ===
st.markdown(
    """
    <div style="text-align: center; margin-top: 10px; margin-bottom: 30px;">
        <div class="card" style="display: inline-block; text-align: justify; max-width: 800px; padding: 20px;">
            <div class="card-title" style="text-align: center;">Tujuan Website</div>
            <div>
                Website ini bertujuan untuk memvisualisasikan hasil pengelompokan (clustering) wilayah di Indonesia 
                berdasarkan indikator status gizi penduduk. 
                Melalui pendekatan ini, pengguna dapat memahami pola dan perbedaan tingkat gizi antar wilayah, 
                serta mengidentifikasi daerah yang memerlukan perhatian lebih dalam peningkatan status gizi. 
                Selain itu, website ini juga menjadi alat bantu analisis bagi peneliti, pembuat kebijakan, 
                maupun masyarakat umum untuk melihat kondisi gizi secara interaktif dan informatif.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")  # spasi

# === Layout 2 atas dan 1 bawah ===
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="card">
            <div class="card-title">Gizi</div>
            <div>
                Gizi merupakan komponen penting dalam menentukan kualitas hidup penduduk suatu wilayah. 
                Status gizi yang baik berkontribusi pada kesehatan, produktivitas, dan kesejahteraan masyarakat.
                Oleh karena itulah penting untuk memahami pola gizi di berbagai wilayah.
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="card">
            <div class="card-title">Metode yang digunakan</div>
            <div>
                Metode clustering yang digunakan untuk mengelompokkan wilayah adalah K-Means, K-Median, dan CLARA (Clustering Large Applications).
            </div>
        </div>
    """, unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
        <div class="card">
            <div class="card-title">K-Means</div>
            <div>
                K-Means adalah algoritma clustering yang membagi data ke dalam beberapa kelompok (cluster) 
                berdasarkan kemiripan nilai-nilai fitur. 
                Algoritma ini mencari pusat cluster yang meminimalkan jarak antar data dalam cluster tersebut.
            </div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="card">
            <div class="card-title">K-Median</div>
            <div>
                K-Median adalah varian dari K-Means yang menggunakan median sebagai pusat cluster. 
                Hal ini membuat K-Median lebih tahan terhadap outlier dibandingkan K-Means, 
                sehingga sering digunakan ketika data memiliki distribusi yang tidak merata.
            </div>
        </div>
    """, unsafe_allow_html=True)

# === Box bawah tengah ===
st.markdown(
    """
    <div style="display: flex; justify-content: center; margin-top: 20px;">
        <div class="card" style="width: 50%;">
            <div class="card-title">CLARA (Clustering Large Applications)</div>
            <div>
                CLARA adalah algoritma clustering yang dirancang untuk menangani dataset besar. 
                Algoritma ini bekerja dengan mengambil sampel dari data dan menerapkan metode K-Medoids pada sampel tersebut. 
                Proses ini diulang beberapa kali untuk menemukan representasi terbaik dari cluster dalam data asli.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
