import json, pathlib
import pandas as pd
import streamlit as st
from app.equity_extract import extract_equity

# Välj motor: backtracker först, annars btwrap
try:
    from app.backtracker import run_backtest
    ENGINE = "backtracker"
except Exception:
    from app import btwrap as W
    run_backtest = W.run_backtest  # type: ignore
    ENGINE = "btwrap"

st.set_page_config(page_title="Backtest V2", page_icon="🧪", layout="wide")
st.title("🧪 Backtest V2")
st.caption(f"Engine: **{ENGINE}**")

def pick_equity_payload(res: dict):
    # Undvik bool-cast på DataFrame – välj första icke-None nyckeln explicit
    for k in ("equity", "summary"):
        if k in res and res[k] is not None:
            return res[k]
    return res

# Lista profilfiler
profs = sorted(pathlib.Path("profiles").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not profs:
    st.warning("Hittar inga filer i profiles/*.json")
    st.stop()

col_left, col_right = st.columns([1,2])
with col_left:
    fname = st.selectbox("Profilfil", [p.name for p in profs])
    data = json.loads((pathlib.Path("profiles")/fname).read_text(encoding="utf-8"))
    plist = data.get("profiles") or []
    pname = st.selectbox("Profil", [p.get("name","<namn saknas>") for p in plist])

pp = next((p for p in plist if p.get("name")==pname), None)
if pp is None:
    st.error("Profil saknas i filen.")
    st.stop()

ticker = pp.get("ticker") or (pp.get("params") or {}).get("ticker")
params = dict(pp.get("params") or {})

with st.spinner(f"Kör backtest för {ticker}…"):
    res = run_backtest(p={"ticker": ticker, "params": params})

x = pick_equity_payload(res)
eq = extract_equity(x)  # -> pd.Series
first = float(eq.iloc[0]) if len(eq) else None
last  = float(eq.iloc[-1]) if len(eq) else None
tr_calc = (last - 1.0) if last is not None else None
facit = (pp.get("metrics") or {}).get("TotalReturn")

with col_right:
    st.subheader(f"{pname} — {ticker}")
    m1,m2,m3 = st.columns(3)
    m1.metric("Equity first", f"{first:.6f}" if first is not None else "–")
    m2.metric("Equity last",  f"{last:.6f}"  if last  is not None else "–")
    m3.metric("TR (calc)",    f"{tr_calc:.6f}" if tr_calc is not None else "–")
    if facit is not None:
        st.metric("TR (facit)", f"{float(facit):.6f}")

    st.line_chart(pd.DataFrame({"Equity": eq.values}, index=eq.index))
