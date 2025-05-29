# main.py
import streamlit as st
import subprocess
import os

st.set_page_config(layout="centered", page_title="Momentum Screener Launcher")
st.title("📊 Momentum Screener Launcher")
st.markdown("Select the screener you want to run:")

screener = st.selectbox("Choose a Screener:", ["-- Select --", "NIFTY 200", "NIFTY 500"])

if st.button("🚀 Run Screener"):
    if screener == "NIFTY 200":
        os.system("streamlit run Momentum_Sector_Breakout_Screener_N200.py")
    elif screener == "NIFTY 500":
        os.system("streamlit run Momentum_Sector_Breakout_Screener.py")
    else:
        st.warning("Please select a valid screener.")
