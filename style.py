import streamlit as st

def load_custom_style():
    full_css = """
    <style>
    /* === Sidebar Styling === */
    [data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        background: linear-gradient(180deg, #1f1f1f 0%, #2b2b2b 100%) !important;
        color: #e0e0e0 !important;
    }

    [data-testid="stSidebarNav"] li a {
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 14px 14px !important;
        margin: 6px 8px !important;
        border-radius: 10px !important;
        background-color: #2b2b2b !important;
        color: #e0e0e0 !important;
        display: block !important;
        text-decoration: none !important;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    [data-testid="stSidebarNav"] li a:not([data-testid="stSidebarNavLink-active"]):hover {
        background-color: #3a3a3a !important;
        color: #1E90FF !important;
        transform: scale(1.03);
    }

    [data-testid="stSidebarNav"] li a[data-testid="stSidebarNavLink-active"] {
        background-color: #4da6ff !important;
        color: white !important;
        font-weight: 700 !important;
    }

    /* === Card Styling === */
    .card {
        background-color: #2e2e2e;
        color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        font-size: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin: 10px; /* ✅ Jarak antar card */
    }

    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 10px rgba(0,0,0,0.4);
    }

    .card-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 12px;
        color: #4da6ff;
    }

    /* === Background app === */
    [data-testid="stAppViewContainer"],
    [data-testid="stToolbar"],
    [data-testid="stHeader"] {
        background: none !important;
    }

    /* === Jarak antar kolom & elemen utama === */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
    """

    st.markdown(full_css, unsafe_allow_html=True)
