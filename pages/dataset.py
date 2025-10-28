import streamlit as st
from style import load_custom_style
from pathlib import Path

st.set_page_config(
    page_title="Dataset - Clustering Wilayah di Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_style()

st.title("📊 Dataset")
st.write("Halaman ini berisi data yang digunakan untuk proses clustering.")
st.markdown(
    """
    <div class="card">
        <div class="card-title">Deskripsi Dataset</div>
        <div>
            Dataset terdiri dari <b>PoU</b>, <b>Jumlah Penduduk</b>, <b>Penduduk Undernourish</b>, dan <b>Persentase Penduduk Miskin P(0)</b> 
            dari total 489 kabupaten di Indonesia pada tahun 2018 sampai 2023.
            Data ini mencakup berbagai indikator gizi yang relevan untuk analisis clustering wilayah di Indonesia.
            <br><br>
            <b>Data ini terdiri dari:</b>
            <ul style="text-align: left; padding-left: 25px; margin-top: 5px;">
                <li><b>Kabupaten:</b> Nama kabupaten di Indonesia.</li>
                <li><b>PoU:</b> Prevalensi undernourish (PoU) penduduk per kabupaten di Indonesia.</li>
                <li><b>Jumlah Penduduk:</b> Total jumlah penduduk per kabupaten di Indonesia.</li>
                <li><b>Penduduk Undernourish:</b> Jumlah penduduk yang mengalami undernourish per kabupaten di Indonesia.</li>
                <li><b>Persentase Penduduk Miskin P(0):</b> Persentase penduduk miskin per kabupaten di Indonesia.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# === Bagian Tombol Download ===
st.markdown("### 📥 Unduh Template & Panduan")

# Buat 3 kolom sejajar
col1, col2, col3 = st.columns(3)

# Path folder template
template_path = Path("template")

# === Tombol 1: Template Dataset ===
template_file = template_path / "Template_Dataset.xlsx"
if template_file.exists():
    with open(template_file, "rb") as f:
        col1.download_button(
            label="📘 Template Dataset",
            data=f,
            file_name="Template_Dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    col1.warning("❌ Template_Dataset.xlsx tidak ditemukan.")

# === Tombol 2: Contoh Data ===
contoh_file = template_path / "Contoh_Data.xlsx"
if contoh_file.exists():
    with open(contoh_file, "rb") as f:
        col2.download_button(
            label="📊 Contoh Data",
            data=f,
            file_name="Contoh_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    col2.warning("❌ Contoh_Data.xlsx tidak ditemukan.")

# === Tombol 3: Buku Panduan ===
panduan_file = template_path / "Buku_Panduan_program.pdf"
if panduan_file.exists():
    with open(panduan_file, "rb") as f:
        col3.download_button(
            label="📗 Buku Panduan",
            data=f,
            file_name="Buku_Panduan_program.pdf",
            mime="application/pdf"
        )
else:
    col3.warning("❌ Buku_Panduan.pptx tidak ditemukan.")
