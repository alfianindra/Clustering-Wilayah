import streamlit as st
import pandas as pd
import numpy as np
import time
import folium
from streamlit.components.v1 import html
from PIL import Image
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from utils import kmeans_manual, kmedian_manual, clara_manual
from io import BytesIO
from matplotlib import cm
from style import load_custom_style
import tempfile
import os

# config halaman
st.set_page_config(
    page_title="Clustering Wilayah di Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_style()

st.markdown("<h1 style='text-align: center;'>Clustering Gizi di Indonesia</h1>", unsafe_allow_html=True)
st.write("""Aplikasi ini melakukan **Clustering Data Gizi** menggunakan algoritma **K-Means**, **K-Median**, dan **CLARA**.""")
st.markdown(
    """
    <div class="card">
        <div class="card-title">⚠️ Perhatian</div>
        <div>
            Sebelum melakukan proses clustering, pastikan Anda sudah mengunduh dan menyiapkan dataset
            pada halaman <a href="/dataset" target="_self" style="color:#4da6ff; font-weight:bold; text-decoration:none;">Dataset</a>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

dataset = st.file_uploader("📂 Unggah Dataset", type=["xlsx", "csv"])
st.markdown("---")

# Parameter cluster
st.subheader("⚙️ Pilih Parameter Clustering")
metode_opsi = [
    "K-Means",
    "K-Median",
    "K-Means dan K-Median",
    "CLARA",
    "K-Means dan K-Median dan CLARA"
]
metode = st.selectbox("Metode Clustering", metode_opsi, index=None, placeholder="Pilih metode")
jumlah_cluster = st.selectbox("Jumlah Cluster (k)", [2, 3, 4, 5, 6, 7], index=None, placeholder="Pilih jumlah cluster")
st.markdown("---")

if "hasil_clustering" not in st.session_state:
    st.session_state.hasil_clustering = None

# Validasi kolom
def validasi_kolom(df):
    required_cols = ["Kabupaten", "Latitude", "Longitude"]
    for year in range(2018, 2024):
        required_cols += [
            f"PoU {year}", f"Jumlah_Penduduk {year}",
            f"Penduduk_Undernourish {year}", f"Persentase Penduduk Miskin (P0) {year}"
        ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Dataset tidak lengkap! Kolom yang hilang: {missing}")
        return False
    return True

# Tampilan peta folium
def tampilkan_peta(df, cluster_col, k, numeric_cols, legend_mode,
                   judul="Peta Hasil Clustering", metode_nama=""):
    st.write(f"### 🗺️ {judul}")
    colors = sns.color_palette("tab10", n_colors=k).as_hex()
    center_lat, center_lon = df["Latitude"].mean(), df["Longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5.8, control_scale=True)

    # Tambahkan marker sesuai cluster
    for _, row in df.iterrows():
        if not np.isfinite(row.get("Latitude", np.nan)) or not np.isfinite(row.get("Longitude", np.nan)):
            continue

        c = int(row[cluster_col])
        color = colors[c % len(colors)]

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            popup=folium.Popup(
                f"<b>{row.get('Kabupaten', '-')}</b><br>Cluster {c}",
                max_width=250
            ),
        ).add_to(m)

    # Buat legend hanya dengan nomor cluster (tanpa keterangan tambahan)
    legend_items = "".join(
        [
            f"<i style='background:{colors[i]}; width:15px; height:15px; display:inline-block; "
            f"margin-right:8px; border-radius:3px;'></i> Cluster {i}<br>"
            for i in range(k)
        ]
    )

    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; width: 180px; background-color: white;
    border:1px solid grey; border-radius:8px; z-index:9999; font-size:13px; padding: 10px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.25);">
        <b>Keterangan Cluster</b><br>{legend_items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    try:
        m.fit_bounds(m.get_bounds())
    except Exception:
        pass

    html(m.get_root().render(), height=720, width="100%")
    return m

# convert peta ke png untuk di download
def Peta_ke_png(m):
    tmp_dir = tempfile.mkdtemp()
    html_path = os.path.join(tmp_dir, "map_temp.html")
    png_path = os.path.join(tmp_dir, "map_screenshot.png")
    m.save(html_path)

    chromedriver_autoinstaller.install()
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=2560,1440")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get("file://" + html_path)
    time.sleep(3)
    driver.save_screenshot(png_path)
    driver.quit()

    with open(png_path, "rb") as f:
        img_bytes = f.read()
    return img_bytes

# PROSES DATASET
if dataset is not None:
    try:
        df = pd.read_csv(dataset) if dataset.name.endswith(".csv") else pd.read_excel(dataset)
        st.success("✅ Dataset berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        st.stop()

    if len(df) < 10:
        st.warning("⚠️ Dataset minimal 10 baris.")
        st.stop()

    if not validasi_kolom(df):
        st.stop()

    exclude_cols = ["Kabupaten", "Latitude", "Longitude"]
    numeric_cols_all = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    st.subheader("📊 Pilih Variabel untuk Analisis")
    pilih_semua = st.checkbox("✅ Pilih semua variabel", value=True)
    pilihan_variabel = st.multiselect("Pilih variabel:", options=numeric_cols_all,
                                      default=numeric_cols_all if pilih_semua else [])
    st.markdown("---")

    # ANALISIS KORELASI
    if pilihan_variabel:
        st.subheader("🔗 Analisis Korelasi antar Variabel Terpilih")

        try:
            tahun_ditemukan = sorted(
                {v.split()[-1] for v in pilihan_variabel if v.split()[-1].isdigit()}
            )

            if len(tahun_ditemukan) == 1:
                tahun_target = tahun_ditemukan[0]
            elif len(tahun_ditemukan) > 1:
                tahun_target = st.selectbox(
                    "Pilih tahun untuk ditampilkan pada heatmap:",
                    tahun_ditemukan,
                    index=0
                )
            else:
                tahun_target = None

            if tahun_target:
                variabel_tahun = [v for v in pilihan_variabel if tahun_target in v]
            else:
                variabel_tahun = pilihan_variabel  

            if not variabel_tahun:
                st.info("Tidak ada variabel dengan tahun yang cocok untuk analisis korelasi.")
            else:
                rename_map = {
                    v: v.replace(f" {tahun_target}", "") if tahun_target else v
                    for v in variabel_tahun
                }
                df_tahun = df[variabel_tahun].rename(columns=rename_map)

                corr = df_tahun.corr(method="pearson")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(
                    corr,
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    fmt=".2f",
                    linewidths=0.5,
                    ax=ax
                )
                ax.set_title(
                    f"Matriks Korelasi Variabel (Data Tahun {tahun_target})"
                    if tahun_target
                    else "Matriks Korelasi Variabel (Tanpa Tahun)",
                    fontsize=11,
                    pad=10
                )
                plt.tight_layout()
                st.pyplot(fig)

                buf_corr = BytesIO()
                fig.savefig(buf_corr, format="png", bbox_inches="tight")
                buf_corr.seek(0)
                st.download_button(
                    label=f"📥 Download Heatmap Korelasi {f'Tahun {tahun_target}' if tahun_target else ''} (PNG)",
                    data=buf_corr,
                    file_name=f"heatmap_korelasi_{tahun_target or 'all'}.png",
                    mime="image/png",
                    key=f"download_corr_heatmap_{tahun_target or 'all'}"
                )

        except Exception as e:
            st.warning(f"Tidak dapat menghitung korelasi: {e}")

    # Tombol Proses clustering
    if st.button("🚀 Lakukan Clustering"):
        if metode not in metode_opsi or jumlah_cluster is None:
            st.warning("⚠️ Pilih metode dan jumlah cluster!")
            st.stop()

        if not pilihan_variabel:
            st.warning("⚠️ Pilih minimal satu variabel!")
            st.stop()

        # Imputasi missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "", inplace=True)

        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[pilihan_variabel] = scaler.fit_transform(df[pilihan_variabel])

        start = time.time()
        result = {}

        # Jika Metode yang dipilih terdapat elemen K-Means
        if metode in ["K-Means", "K-Means dan K-Median", "K-Means dan K-Median dan CLARA"]:
            labels_kmeans, _ = kmeans_manual(df_scaled[pilihan_variabel], jumlah_cluster)
            result["K-Means"] = {
                "labels": labels_kmeans,
                "silhouette": silhouette_score(df_scaled[pilihan_variabel], labels_kmeans),
                "dbi": davies_bouldin_score(df_scaled[pilihan_variabel], labels_kmeans)
            }

        # Jika Metode yang dipilih terdapat elemen K-Median 
        if metode in ["K-Median", "K-Means dan K-Median", "K-Means dan K-Median dan CLARA"]:
            labels_kmedian, _ = kmedian_manual(df_scaled[pilihan_variabel], jumlah_cluster)
            result["K-Median"] = {
                "labels": labels_kmedian,
                "silhouette": silhouette_score(df_scaled[pilihan_variabel], labels_kmedian),
                "dbi": davies_bouldin_score(df_scaled[pilihan_variabel], labels_kmedian)
            }

        # Jika Metode yang dipilih terdapat elemen CLARA 
        if metode in ["CLARA", "K-Means dan K-Median dan CLARA"]:
            labels_clara, _ = clara_manual(df_scaled[pilihan_variabel], jumlah_cluster)
            result["CLARA"] = {
                "labels": labels_clara,
                "silhouette": silhouette_score(df_scaled[pilihan_variabel], labels_clara),
                "dbi": davies_bouldin_score(df_scaled[pilihan_variabel], labels_clara)
            }

        result["waktu"] = time.time() - start
        result["data"] = df
        result["scaled"] = df_scaled
        result["numeric_cols"] = pilihan_variabel
        st.session_state.hasil_clustering = (metode, jumlah_cluster, result)

# Tampilan Hasil
if st.session_state.hasil_clustering:
    metode, jumlah_cluster, result = st.session_state.hasil_clustering
    df = result["data"].copy()
    df_scaled = result["scaled"]
    numeric_cols = result["numeric_cols"]

    st.success(f"✅ Clustering selesai ({metode}, {jumlah_cluster} cluster)")

    def tampil_hasil(df, labels, metode_nama):
        df = df.copy()
        df["Cluster"] = labels
        kolom_tampil = [c for c in ["Kabupaten", "Latitude", "Longitude"] if c in df.columns]
        kolom_tampil += numeric_cols + ["Cluster"]
        df_tampil = df[kolom_tampil]

        st.write(f"### 📊 Hasil Clustering ({metode_nama})")
        st.dataframe(df_tampil)

        #download Tabel csv
        csv_buffer = BytesIO()
        df_tampil.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            label=f"⬇️ Download Hasil Clustering {metode_nama} (CSV)",
            data=csv_buffer,
            file_name=f"hasil_clustering_{metode_nama.lower().replace(' ','_')}.csv",
            mime="text/csv",
            key=f"download_csv_{metode_nama}"
        )

        # Tampilan BOX PLOT 
        st.markdown("### Distribusi Variabel per Cluster")

        @st.cache_data(show_spinner=False)
        def generate_boxplots(df, numeric_cols):
            """Generate semua boxplot sekaligus (cached, grid 3 kolom per baris, no flicker)"""
            all_buffers = []
            imgs = []

            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(3.8, 2.6))
                try:
                    sns.boxplot(x="Cluster", y=col, data=df, palette="tab10", ax=ax)
                except Exception:
                    ax.boxplot([df[df["Cluster"] == c][col].dropna().values for c in sorted(df["Cluster"].unique())])
                    ax.set_xticklabels([f"Cluster {c}" for c in sorted(df["Cluster"].unique())])
                ax.set_title(col, fontsize=9, fontweight="bold", pad=6)
                ax.set_xlabel("Cluster", fontsize=8)
                ax.set_ylabel("Nilai", fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
                plt.tight_layout()

                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                buf.seek(0)
                all_buffers.append(buf)
                imgs.append(Image.open(buf))
                plt.close(fig)

            cols_per_row = 3
            padding = 10

            rows = (len(imgs) + cols_per_row - 1) // cols_per_row
            img_width = max(i.width for i in imgs)
            img_height = max(i.height for i in imgs)
            canvas_width = cols_per_row * img_width + (cols_per_row - 1) * padding
            canvas_height = rows * img_height + (rows - 1) * padding

            combined = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
            for idx, im in enumerate(imgs):
                row = idx // cols_per_row
                col = idx % cols_per_row
                x = col * (img_width + padding)
                y = row * (img_height + padding)
                combined.paste(im, (x, y))

            out_buf = BytesIO()
            combined.save(out_buf, format="PNG", optimize=True)
            out_buf.seek(0)
            return all_buffers, out_buf

        all_boxplots, buf_all = generate_boxplots(df, numeric_cols)

        cols_layout = st.columns(3)
        for i, buf in enumerate(all_boxplots):
            with cols_layout[i % 3]:
                st.image(buf, use_container_width=True)

        st.download_button(
            label=f"📥 Download Semua Boxplot ({metode_nama})",
            data=buf_all,
            file_name=f"boxplot_{metode_nama.lower()}.png",
            mime="image/png",
            key=f"download_boxplot_{metode_nama}"
        )

        # PETA 
        m = tampilkan_peta(df, "Cluster", jumlah_cluster, numeric_cols, "auto",
                           f"Peta Hasil Clustering ({metode_nama})", metode_nama)

        # Proses Peta
        if st.button(f"💾 Simpan Peta ({metode_nama})", key=f"save_map_btn_{metode_nama}"):
            with st.spinner("💾 Sedang menyiapkan file peta, mohon tunggu..."):
                try:
                    # Ambil representasi HTML dari objek folium
                    html_data = m.get_root().render()
                    st.session_state[f"html_peta_{metode_nama}"] = html_data.encode("utf-8")
                    st.success("✅ Peta berhasil disiapkan untuk diunduh!")
                except Exception as e:
                    st.error(f"❌ Gagal menyimpan peta: {e}")
                    st.session_state.pop(f"html_peta_{metode_nama}", None)

        # Tampilkan tombol download setelah selesai proses
        if f"html_peta_{metode_nama}" in st.session_state:
            try:
                st.download_button(
                    label=f"⬇️ Unduh Peta ({metode_nama}) (HTML)",
                    data=st.session_state[f"html_peta_{metode_nama}"],
                    file_name=f"peta_{metode_nama.lower()}.html",
                    mime="text/html",
                    key=f"download_html_map_{metode_nama}"
                )
            except Exception as e:
                st.error(f"⚠️ Gagal menyiapkan tombol unduh: {e}")

        # TREN VARIABEL  
        st.markdown("### 📈 Tren Variabel Tiap Cluster")

        @st.cache_data(show_spinner=False)
        def generate_trends_static(df, numeric_cols, cols_per_row=3):
            tahun_cols = [str(y) for y in range(2018, 2024)]
            variabel_tahunan = [v for v in numeric_cols if any(t in v for t in tahun_cols)]
            all_buffers, imgs = [], []

            if not variabel_tahunan:
                return [], None

            bases = sorted(set(" ".join(v.split()[:-1]) if v.split()[-1].isdigit() else v for v in variabel_tahunan))

            for base in bases:
                kolom_tahun = [v for v in variabel_tahunan if (v.startswith(base) or base in v)]
                kolom_tahun = [v for v in kolom_tahun if v.split()[-1].isdigit()]
                if not kolom_tahun:
                    continue

                kolom_tahun_sorted = sorted(kolom_tahun, key=lambda x: int(x.split()[-1]))
                df_tren = df.groupby("Cluster")[kolom_tahun_sorted].mean().T
                df_tren.index = [int(c.split()[-1]) for c in kolom_tahun_sorted]

                fig, ax = plt.subplots(figsize=(3.8, 2.4))
                for c in df_tren.columns:
                    ax.plot(df_tren.index, df_tren[c], marker="o", label=f"Cluster {c}")

                ax.set_title(base, fontsize=9)
                ax.set_xlabel("Tahun", fontsize=8)
                ax.set_ylabel("Rata-rata", fontsize=8)

                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.80, box.height])  # sisakan ruang kanan 20%
                ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

                ax.tick_params(axis='both', labelsize=8)
                plt.tight_layout()

                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                buf.seek(0)
                all_buffers.append(buf)
                imgs.append(Image.open(buf))
                plt.close(fig)

            if not imgs:
                return all_buffers, None

            max_w = max(im.width for im in imgs)
            max_h = max(im.height for im in imgs)
            n_cols = cols_per_row
            n_rows = (len(imgs) + n_cols - 1) // n_cols

            combined_width = n_cols * max_w
            combined_height = n_rows * max_h
            combined_tren = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))

            for idx, im in enumerate(imgs):
                row = idx // n_cols
                col = idx % n_cols
                x_offset = col * max_w
                y_offset = row * max_h
                combined_tren.paste(im, (x_offset, y_offset))

            buf_tren = BytesIO()
            combined_tren.save(buf_tren, format="PNG", optimize=True)
            buf_tren.seek(0)

            return all_buffers, buf_tren

        all_trends, buf_tren = generate_trends_static(df, numeric_cols)

        if all_trends:
            st.markdown("#### 🔹 Visualisasi Tren per Variabel")
            cols_trend_layout = st.columns(2)
            for i, buf in enumerate(all_trends):
                with cols_trend_layout[i % 2]:
                    st.image(buf, use_container_width=True)

            st.download_button(
                label=f"📥 Download Semua Grafik Tren ({metode_nama})",
                data=buf_tren,
                file_name=f"trend_{metode_nama.lower()}.png",
                mime="image/png",
                key=f"download_trend_{metode_nama}"
            )
        else:
            st.info("Tidak ada variabel tahunan (2018–2023) untuk ditampilkan.")

        #TREN 10 WILAYAH TERATAS
        st.markdown("### Tren Tahunan 10 Wilayah Teratas")

        @st.cache_data(show_spinner=False)
        def generate_top10_trend(df, numeric_cols):
            tahun_cols = [str(y) for y in range(2018, 2024)]
            variabel_tahunan = [v for v in numeric_cols if any(t in v for t in tahun_cols)]
            all_buffers, imgs = [], []

            if not variabel_tahunan:
                return [], None

            bases = sorted(set(" ".join(v.split()[:-1]) if v.split()[-1].isdigit() else v for v in variabel_tahunan))

            for base in bases:
                kolom_tahun = [v for v in variabel_tahunan if (v.startswith(base) or base in v)]
                kolom_tahun = [v for v in kolom_tahun if v.split()[-1].isdigit()]
                if not kolom_tahun:
                    continue

                kolom_tahun_sorted = sorted(kolom_tahun, key=lambda x: int(x.split()[-1]))

                df['mean_base'] = df[kolom_tahun_sorted].mean(axis=1)
                df_top10 = df.nlargest(10, 'mean_base')
                df_top10 = df_top10[["Kabupaten"] + kolom_tahun_sorted].set_index("Kabupaten")

                fig, ax = plt.subplots(figsize=(5, 3))
                for kab, row in df_top10.iterrows():
                    ax.plot(
                        [int(c.split()[-1]) for c in kolom_tahun_sorted],
                        row.values,
                        marker="o",
                        label=kab
                    )

                ax.set_title(f"{base} (10 Wilayah Teratas)", fontsize=9, fontweight="bold")
                ax.set_xlabel("Tahun", fontsize=8)
                ax.set_ylabel("Nilai", fontsize=8)
                
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.80, box.height])  # sisakan ruang kanan 20%
                ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

                ax.tick_params(axis='both', labelsize=8)
                plt.tight_layout()


                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                buf.seek(0)
                all_buffers.append(buf)
                imgs.append(Image.open(buf))
                plt.close(fig)

            if not imgs:
                return all_buffers, None

            cols_per_row = 2
            max_w = max(im.width for im in imgs)
            max_h = max(im.height for im in imgs)
            n_rows = (len(imgs) + cols_per_row - 1) // cols_per_row
            combined_width = cols_per_row * max_w
            combined_height = n_rows * max_h
            combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))

            for idx, im in enumerate(imgs):
                row = idx // cols_per_row
                col = idx % cols_per_row
                combined_img.paste(im, (col * max_w, row * max_h))

            buf_combined = BytesIO()
            combined_img.save(buf_combined, format="PNG", optimize=True)
            buf_combined.seek(0)

            return all_buffers, buf_combined

        all_top10, buf_top10 = generate_top10_trend(df, numeric_cols)

        if all_top10:
            st.markdown("#### 🔹 Visualisasi Tren 10 Wilayah Teratas per Variabel")
            cols_top_layout = st.columns(2)
            for i, buf in enumerate(all_top10):
                with cols_top_layout[i % 2]:
                    st.image(buf, use_container_width=True)

            st.download_button(
                label=f"📥 Download Semua Tren 10 Wilayah Teratas ({metode_nama})",
                data=buf_top10,
                file_name=f"top10_tren_{metode_nama.lower()}.png",
                mime="image/png",
                key=f"download_top10_{metode_nama}"
            )
        else:
            st.info("Tidak ada variabel tahunan untuk ditampilkan pada tren 10 wilayah teratas.")
        # Grafik nilai silhouette per cluster
        st.markdown("### 📈 Silhouette Plot per Cluster")

        try:

            data_matriks = df_scaled[numeric_cols].values
            label_cluster = labels
            algo = metode_nama

            nilai_sample = silhouette_samples(data_matriks, label_cluster)
            nilai_rata   = silhouette_score(data_matriks, label_cluster)

            n_clusters = len(np.unique(label_cluster))
            y_bawah = 5

            fig, ax1 = plt.subplots(figsize=(5, 3.5))
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
            unique_labels = sorted(np.unique(label_cluster))
            color_map = {label: colors[i % 10] for i, label in enumerate(unique_labels)}
            for i in unique_labels:
                nilai_i = nilai_sample[label_cluster == i]
                nilai_i.sort()

                ukuran_i = nilai_i.shape[0]
                y_atas = y_bawah + ukuran_i

                warna = color_map[i]
                ax1.fill_betweenx(
                    np.arange(y_bawah, y_atas),
                    0,
                    nilai_i,
                    facecolor=warna,
                    edgecolor=warna,
                    alpha=0.7
                )

                ax1.text(
                    -0.25, 
                    y_bawah + 0.5 * ukuran_i,
                    str(i),
                    fontsize=9,
                    va='center',
                    ha='right'
                )
                y_bawah = y_atas + 5

            ax1.axvline(x=nilai_rata, color="red", linestyle="--", linewidth=1)

            ax1.set_yticks([])  
            ax1.set_xlim([-0.3, 1])  
            ax1.set_title(f"Plot Silhouette ({algo})", fontsize=10, pad=8)
            ax1.set_xlabel("Nilai Silhouette Coefficient", fontsize=9)
            ax1.set_ylabel("Cluster", fontsize=9)
            plt.tight_layout(pad=0.8)

            st.pyplot(fig)

            # Tombol download
            buf_sil = BytesIO()
            fig.savefig(buf_sil, format="png", bbox_inches="tight", dpi=150)
            buf_sil.seek(0)
            st.download_button(
                label=f"📥 Download Grafik Silhouette ({algo})",
                data=buf_sil,
                file_name=f"silhouette_plot_{algo.lower()}.png",
                mime="image/png",
                key=f"download_silhouette_{algo}"
            )

        except Exception as e:
            st.warning(f"⚠️ Gagal membuat grafik silhouette: {e}")

        # METRIK  
        col1, col2, col3 = st.columns(3)
        silh = result.get(metode_nama, {}).get("silhouette", None)
        dbi = result.get(metode_nama, {}).get("dbi", None)
        waktu = result.get("waktu", None)
        col1.metric("Silhouette Score", f"{silh:.4f}" if silh is not None else "N/A")
        col2.metric("Davies-Bouldin", f"{dbi:.4f}" if dbi is not None else "N/A")
        col3.metric("Waktu Komputasi", f"{waktu:.4f} detik" if waktu is not None else "N/A")

    # Jalankan semua hasil sesuai metode
    if metode in ["K-Means", "K-Means dan K-Median", "K-Means dan K-Median dan CLARA"]:
        tampil_hasil(result["data"].copy(), result["K-Means"]["labels"], "K-Means")

    if metode in ["K-Median", "K-Means dan K-Median", "K-Means dan K-Median dan CLARA"]:
        tampil_hasil(result["data"].copy(), result["K-Median"]["labels"], "K-Median")

    if metode in ["CLARA", "K-Means dan K-Median dan CLARA"]:
        tampil_hasil(result["data"].copy(), result["CLARA"]["labels"], "CLARA")
