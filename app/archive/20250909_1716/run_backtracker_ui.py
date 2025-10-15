from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Dalatrader – Backtest",
    page_icon="📈",
    layout="wide",
)

OUT_DIR = Path("./outputs").resolve()
PROFILES_FP = OUT_DIR / "profiles" / "backtest_profiles.json"

st.sidebar.title("📚 Sidor")
st.sidebar.markdown("- **Backtest** (vänster meny)")

# Miljökoll
bd_ok = bool(os.environ.get("BORSDATA_KEY"))
st.sidebar.markdown("---")
st.sidebar.subheader("Miljökontroll")
st.sidebar.write(f"BORSDATA_KEY: {'✅' if bd_ok else '❌'}")
st.sidebar.write(f"Outputs: `{OUT_DIR}`")

st.title("Dalatrader – Backtest & profiler")
st.write(
    "Använd sidmenyn **Backtest** för att köra tester, spara/ladda profiler och exportera resultat."
)

with st.expander("Var sparas profiler?"):
    st.code(str(PROFILES_FP), language="text")

