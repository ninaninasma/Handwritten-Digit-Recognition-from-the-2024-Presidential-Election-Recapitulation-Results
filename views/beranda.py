import streamlit as st
import base64
from config import LOGO_PATH

# ======================================================
# UTIL
# ======================================================
def load_logo_base64():
    with open(LOGO_PATH, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# ======================================================
# RENDER
# ======================================================
def render():
    logo_base64 = load_logo_base64()

    # ==================== STYLE (TER-SCOPE AMAN) ==================== #
    st.markdown("""
    <style>
        .beranda-container {
            text-align: center;
            padding: 2rem 1rem;
            background-color: #F5F5F5;
        }

        .beranda-logo {
            width: 420px;
            margin-bottom: 1rem;
        }

        .beranda-subtitle {
            color: #555;
            margin-bottom: 2rem;
            font-size: 1.2rem;
        }

        /* ===== INFO BOX ===== */
        .info-box {
            max-width: 900px;
            margin: 2rem auto;
            background: #FFF5F5;
            border-left: 6px solid #C62828;
            border-radius: 12px;
            padding: 1.6rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: left;
        }

        .info-title {
            color: #C62828;
            margin-top: 0;
            font-size: 1.25rem;
            font-weight: 700;
        }

        /* ===== CTA BOX ===== */
        .cta-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 2.5rem auto;
            background: #FFF0F0;
            border: 1px solid #F2B8B8;
            border-radius: 14px;
            padding: 1.6rem;
            max-width: 520px;
        }

        .cta-text {
            font-size: 1.1rem;
            color: #333;
            font-weight: 600;
            text-align: center;
            margin-bottom: 1rem;
        }

        /* ===== CTA BUTTON ONLY (AMAN, TIDAK SENTUH SIDEBAR) ===== */
        .cta-container div.stButton > button {
            background-color: #C62828 !important;
            color: #FFFFFF !important;
            border-radius: 10px;
            padding: 0.6rem 2rem;
            font-weight: 700;
            font-size: 1rem;
            border: none;
            box-shadow: 0 4px 10px rgba(198,40,40,0.25);
        }

        .cta-container div.stButton > button:hover {
            background-color: #A61B1B !important;
            box-shadow: 0 6px 14px rgba(198,40,40,0.35);
        }

        .cta-container div.stButton > button:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(198,40,40,0.3);
        }
    </style>
    """, unsafe_allow_html=True)

    # ==================== HEADER ==================== #
    st.markdown(f"""
    <div class="beranda-container">
        <img src="data:image/png;base64,{logo_base64}" class="beranda-logo" />
        <div class="beranda-subtitle">
            Pengenalan Digit Tulisan Tangan<br/>
            Rekapitulasi Pemilihan Presiden 2024
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==================== INFO ==================== #
    st.markdown("""
    <div class="info-box">
        <h4 class="info-title">Tentang PemiluDigit</h4>
        <p>
        PemiluDigit merupakan aplikasi berbasis kecerdasan buatan yang dirancang
        untuk membantu proses digitalisasi data hasil rekapitulasi pemungutan suara
        pada Pemilihan Presiden tahun 2024.
        </p>
        <p>
        Sistem ini berfokus pada <strong>pengenalan digit numerik</strong> dari
        tulisan tangan pada formulir rekap dan tidak melakukan interpretasi politik
        maupun penentuan hasil pemilihan.
        </p>
        <p>
        Dengan pendekatan pemrosesan citra dan pembelajaran mesin,
        PemiluDigit mendukung efisiensi, konsistensi, dan transparansi
        dalam pengolahan data rekapitulasi.
        </p>
        <p style="font-style: italic; color: #666;">
        Aplikasi ini bersifat netral dan digunakan sebagai alat bantu teknis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==================== CTA ==================== #
    st.markdown("""
    <div class="cta-container">
        <div class="cta-text">
            Mulai proses pengenalan digit dari formulir rekap pemilu
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("MULAI UJI DIGIT"):
            st.session_state.current_page = "Uji Digit"
            st.rerun()
