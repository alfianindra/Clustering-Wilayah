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

# === Load CSS utama ===
load_custom_style()

# === Judul halaman ===
st.markdown("<h1 style='text-align: center;'>PROFILE</h1>", unsafe_allow_html=True)

# === Path gambar profil ===
image_path = Path("image/Profile.jpg")

# === Styling tambahan ===
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

# === Dua card menyamping (Hobi & Latar Belakang) ===
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
