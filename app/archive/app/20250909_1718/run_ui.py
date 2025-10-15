from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Dalatrader UI", layout="wide")

ROOT = Path(__file__).resolve().parent

st.title("Dalatrader – UI")
st.write("Välj sida i menyn till vänster: **Backtrack** eller **Portfölj**.")

st.markdown(
"""
**Tips**
- Backtrack-sidan kan spara **profiler** till `outputs/profiles/*.json`.
- Portfölj-sidan läser dessa profiler och kör portfölj-backtest med dina regler.
"""
)
