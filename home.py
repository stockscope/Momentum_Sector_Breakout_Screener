# Home.py
import streamlit as st

st.set_page_config(page_title="Momentum Screener", layout="centered")

# 🔧 Auto-expand sidebar on mobile using CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        @media (max-width: 768px) {
            [data-testid="stSidebar"] {
                visibility: visible !important;
                width: 260px !important;
                min-width: 260px !important;
                transform: none !important;
                position: relative !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# 🏠 Main title and instructions
st.title("📊 Momentum Screener Dashboard")
st.markdown("Use the sidebar to choose between NIFTY 200 and NIFTY 500 screeners.")

# 📱 Extra hint for mobile users
st.info("📱 On mobile? Tap the menu icon (☰ or ⬅️) in the top-left if you don't see the options.")
