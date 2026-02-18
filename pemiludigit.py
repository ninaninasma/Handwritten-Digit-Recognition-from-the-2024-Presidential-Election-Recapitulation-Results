import streamlit as st
from views import beranda, latih, uji
import base64

# ==================== SESSION STATE ==================== #
if "current_page" not in st.session_state:
    st.session_state.current_page = "Beranda"

# ==================== LOAD LOGO ==================== #
def load_logo_base64():
    with open("pemilu.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_logo_base64()

# ==================== CSS ==================== #
st.markdown("""
<style>
    :root {
        --red-primary: #C62828;
        --white: #FFFFFF;
        --gray-light: #F5F5F5;
        --gray-border: #DADADA;
        --text-dark: #2E2E2E;
        --text-light: #FFFFFF;
    }

    .stApp {
        background-color: var(--gray-light);
    }

    [data-testid="stSidebar"] {
        background-color: var(--white);
        padding: 0.6rem;
        border-right: 1px solid var(--gray-border);
    }

    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding-bottom: 0.6rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--red-primary);
    }

    .sidebar-logo {
        width: 80px;
    }

    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--red-primary);
        line-height: 1.2;
        text-align: center;
    }

    .sidebar-button-container {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
    }

    .stButton > button {
        width: 100%;
        height: 42px;
        background-color: var(--white);
        color: var(--text-dark);
        border: 1px solid var(--gray-border);
        border-radius: 8px;
        font-weight: 600;
        text-align: left;
        padding-left: 1rem;
    }

    .stButton > button[kind="primary"] {
        background-color: var(--red-primary);
        color: var(--text-light);
        border: none;
    }

    .stButton > button:not([kind="primary"]):hover {
        background-color: #EEEEEE;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ==================== #
def create_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-header">
            <img src="data:image/png;base64,{logo_base64}" class="sidebar-logo" />
            <div class="sidebar-title">
                PENGENALAN DIGIT<br/>
                REKAP PEMILU 2024
            </div>
        </div>
        """, unsafe_allow_html=True)

        menu = ["Beranda", "Latih Model", "Uji Digit"]

        st.markdown('<div class="sidebar-button-container">', unsafe_allow_html=True)

        for item in menu:
            if st.button(
                item,
                key=f"menu_{item}",
                type="primary" if st.session_state.current_page == item else "secondary"
            ):
                st.session_state.current_page = item
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# ==================== MAIN ==================== #
def render():
    create_sidebar()

    if st.session_state.current_page == "Beranda":
        beranda.render()
    elif st.session_state.current_page == "Latih Model":
        latih.render()
    elif st.session_state.current_page == "Uji Digit":
        uji.render()

# ==================== ENTRY ==================== #
if __name__ == "__main__":
    render()
