import streamlit as st

def load_custom_style():
    full_css = """
    <style>
    /* ==============================
       🎨 TEMA ADAPTIF (Light & Dark)
    ===============================*/

    /* Sidebar */
    [data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        background: var(--sidebar-bg, linear-gradient(180deg, #1f1f1f 0%, #2b2b2b 100%)) !important;
        color: var(--sidebar-text, #e0e0e0) !important;
    }

    [data-testid="stSidebarNav"] li a {
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 14px 14px !important;
        margin: 6px 8px !important;
        border-radius: 10px !important;
        background-color: var(--sidebar-link-bg, #2b2b2b) !important;
        color: var(--sidebar-text, #e0e0e0) !important;
        display: block !important;
        text-decoration: none !important;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    [data-testid="stSidebarNav"] li a:not([data-testid="stSidebarNavLink-active"]):hover {
        background-color: var(--sidebar-hover-bg, #3a3a3a) !important;
        color: var(--accent, #1E90FF) !important;
        transform: scale(1.03);
    }

    [data-testid="stSidebarNav"] li a[data-testid="stSidebarNavLink-active"] {
        background-color: var(--accent, #4da6ff) !important;
        color: white !important;
        font-weight: 700 !important;
    }

    /* === Card Styling === */
    .card {
        background-color: var(--card-bg, #2e2e2e);
        color: var(--card-text, white);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        font-size: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin: 10px;
    }

    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 10px rgba(0,0,0,0.4);
    }

    .card-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 12px;
        color: var(--accent, #4da6ff);
    }

    /* === Background app === */
    [data-testid="stAppViewContainer"],
    [data-testid="stToolbar"],
    [data-testid="stHeader"] {
        background: none !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* ==============================
       🌞 Light Mode
    ===============================*/
    @media (prefers-color-scheme: light) {
        :root {
            --sidebar-bg: linear-gradient(180deg, #f7f7f7 0%, #eaeaea 100%);
            --sidebar-text: #333;
            --sidebar-link-bg: #f0f0f0;
            --sidebar-hover-bg: #dcdcdc;
            --accent: #1E90FF;
            --card-bg: #ffffff;
            --card-text: #000000;
        }
    }

    /* ==============================
       🌚 Dark Mode
    ===============================*/
    @media (prefers-color-scheme: dark) {
        :root {
            --sidebar-bg: linear-gradient(180deg, #1f1f1f 0%, #2b2b2b 100%);
            --sidebar-text: #e0e0e0;
            --sidebar-link-bg: #2b2b2b;
            --sidebar-hover-bg: #3a3a3a;
            --accent: #4da6ff;
            --card-bg: #2e2e2e;
            --card-text: #ffffff;
        }
    }
    </style>
    """

    st.markdown(full_css, unsafe_allow_html=True)
