import streamlit as st
from style import load_custom_style
from pathlib import Path
from base64 import b64encode

# === Konfigurasi halaman ===
st.set_page_config(
    page_title="Profile - Clustering Wilayah di Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)


load_custom_style()

st.markdown("<h1 style='text-align: center;'>PROFILE</h1>", unsafe_allow_html=True)

image_path = Path("image/Profile.jpg")

st.markdown("""
    <style>
    .profile-container {
        text-align: center;
        margin-top: 30px;
    }

    .profile-circle {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        overflow: hidden;
        display: inline-block;
        background-color: #333;
        border: 4px solid #4da6ff;
        box-shadow: 0 0 20px rgba(77,166,255,0.5);
        transition: all 0.3s ease;
    }

    .profile-circle:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(77,166,255,0.8);
    }

    .profile-name {
        color: white;
        font-size: 20px;
        font-weight: bold;
        margin-top: 12px;
    }

    .card {
        background-color: #2e2e2e;
        color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: justify;
        font-size: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-top: 20px;
    }

    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 10px rgba(0,0,0,0.4);
    }

    .card-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #4da6ff;
        text-align: center;
    }

    .usage-section {
        background-color: #1f1f1f;
        padding: 25px;
        margin-top: 35px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: white;
    }

    </style>
""", unsafe_allow_html=True)

# === Tampilkan foto profil ===
if image_path.exists():
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        encoded = b64encode(img_bytes).decode()
        img_base64 = f"data:image/jpeg;base64,{encoded}"

    st.markdown(f"""
        <div class="profile-container">
            <div class="profile-circle">
                <img src="{img_base64}" alt="Foto Profil" width="100%" height="100%" style="object-fit: cover;">
            </div>
            <div class="profile-name">Alfian Indrajaya</div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Gambar profil tidak ditemukan di folder 'image/'. Pastikan nama filenya 'Profile.jpg'.")


col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="card">
            <div class="card-title">🎮 Hobi</div>
            <div>
                Saya memiliki beberapa hobi yang saya lakukan di waktu luang, yaitu bermain game, menonton film, 
                dan mendengarkan musik. Hobi-hobi ini membantu saya untuk bersantai dan mendapatkan inspirasi baru.
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="card">
            <div class="card-title">📘 Latar Belakang</div>
            <div>
                Halo, nama saya <b>Alfian Indrajaya</b>. Saya merupakan mahasiswa Universitas Tarumanagara 
                jurusan <b>Teknik Informatika</b> angkatan 2022. Saya memiliki minat besar di bidang data science 
                dan machine learning, serta senang mengerjakan proyek berbasis analisis data seperti clustering wilayah.
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="usage-section">
        <h2 style="text-align:center; color:#4da6ff;">🧭 Cara Penggunaan Website</h2>
        <p>
            Halaman website ini dirancang untuk membantu Anda menjelajahi hasil analisis clustering wilayah di Indonesia berdasarkan status gizi penduduk. Berikut cara penggunaannya:
        </p>
        <ul>
            <li><b>📁 Download Buku Panduan dan dataset:</b> Pada halaman dataset.</li>
            <li><b>🔍 Eksplorasi Hasil</b> didapatkan ketika sudah mengunggah dataset, memilih metode K-Means, K-Median atau CLARA dan memilih jumlah cluster K lalu tekan tombol lakukan clustering.</li>
            <li><b>🔎 Hasil Clustering:</b> hasilnya berupa boxplot, peta, tren 10 wilayah teratas berdasarkan dataset,dan metrik evaluasi metode yang dipilih.</li>
            <li><b>📥 Download Output:</b> Jika tersedia, user dapat mengunduh hasil analisis dalam format CSV,PNG dan HTML.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="display: flex; justify-content: center; margin-top: 25px;">
        <div class="card" style="width: 60%; text-align: center;">
            <div class="card-title">📞 Kontak</div>
            <div>
                Jika Anda memiliki pertanyaan, saran, atau ingin berdiskusi lebih lanjut mengenai proyek ini, 
                Anda dapat menghubungi saya melalui kontak berikut:
                <br><br>
                📧 <b>Email:</b> <a href="mailto:alfianij8@gmail.com" style="color:#4da6ff;">alfianindrajaya@gmail.com</a><br>
                🧑‍💻 <b>GitHub:</b> <a href="https://github.com/alfianindra" target="_blank" style="color:#4da6ff;">github.com/alfianindrajaya</a>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
