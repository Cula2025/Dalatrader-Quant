from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Trader Demo", page_icon="📈", layout="wide")

st.markdown("""
<style>
:root { --primary: #1f6feb; --accent: #d29922; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Trader Demo – Startsida")
st.write("Välj en sida i menyn till vänster eller öppna backtest direkt:")

try:
    st.page_link("pages/1_Backtrack.py", label="🔁 Öppna: Backtrack – En aktie", icon="↗️")
except Exception:
    st.caption("Öppna backtestsidan via sidomenyn i stället.")