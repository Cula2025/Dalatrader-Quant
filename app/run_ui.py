from pathlib import Path
import streamlit as st

# Ladda .env och mappa nycklar
from app.env_bootstrap import load_env
load_env()

st.set_page_config(page_title="Dalatrader UI", layout="wide")

st.title("Dalatrader – UI")
st.write("Välj sida i menyn till vänster: **Backtrack** eller **Portfölj**.")

st.markdown(
"""
**Tips**
- Skapa/spara profiler på **Backtrack** (t.ex. `outputs/profiles/profiles_demo.json`).
- Kör portfölj med dessa profiler på **Portfölj**.
"""
)

