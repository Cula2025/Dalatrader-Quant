from __future__ import annotations
import json, datetime as dt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtester V3", page_icon="🧪", layout="wide")
st.title("🧪 Backtester V3")

from app_v3.data_provider_v3 import get_ohlcv
from app_v3.bt_core_v3 import run_backtest

# --- Sidebar / Inmatning ---
colL, colR = st.columns([1,1])
with colL:
    ticker = st.text_input("Ticker", value="GETI B")
    fivey = st.checkbox("5 år bakåt (auto)", value=True)
    if fivey:
        end = dt.date.today()
        start = end.replace(year=end.year-5)
    else:
        start = st.date_input("Start", value=dt.date(2020,10,14))
        end = st.date_input("Slut", value=dt.date.today())
with colR:
    default_params = {
        "fast": 10, "slow": 30,
        "use_rsi_filter": False, "rsi_window": 14, "rsi_min": 30.0, "rsi_max": 70.0,
        "fee_bps": 0, "slippage_bps": 0
    }
    params_txt = st.text_area("Parametrar (JSON)", value=json.dumps(default_params, indent=2), height=220)
    try:
        params = json.loads(params_txt) if params_txt.strip() else default_params
        if not isinstance(params, dict): raise ValueError("Params måste vara ett JSON-objekt")
    except Exception as e:
        st.error(f"Ogiltig JSON: {e}")
        st.stop()

if st.button("Kör backtest", type="primary"):
    try:
        df = get_ohlcv(ticker, start=start.isoformat(), end=end.isoformat())
    except Exception as e:
        st.error(f"Fel vid hämtning av OHLCV: {e}")
        st.stop()

    if df.empty:
        st.warning("Ingen data hittades för intervallet.")
        st.stop()

    try:
        res = run_backtest(df, params)
    except Exception as e:
        st.exception(e)
        st.stop()

    # Equity & BH
    eq = pd.to_numeric(pd.Series(res.get("equity")), errors="coerce").dropna()
    px = pd.to_numeric(df["Close"], errors="coerce").dropna()
    px = px.reindex(eq.index, method="ffill").dropna()
    bh = (px / float(px.iloc[0])).rename("BH")

    # Nyckeltal
    btx = float(eq.iloc[-1]/eq.iloc[0]) if len(eq) else float("nan")
    bhx = float(bh.iloc[-1]) if len(bh) else float("nan")
    st.subheader("Resultat")
    m1, m2, m3 = st.columns(3)
    m1.metric("BT× (Strategy / start)", f"{btx:.4f}")
    m2.metric("BH× (Indexerat / start)", f"{bhx:.4f}")
    m3.metric("Antal datapunkter", f"{len(eq):,}")

    # Plott
    plot_df = pd.DataFrame({"Strategy": eq.values, "BH": bh.values}, index=eq.index)

# --- Efter körning: nyckeltal ---
if 'res' in locals() and isinstance(res, dict):
    import numpy as np
    eq = res.get("equity")
    bh = res.get("bh")
    tr = res.get("trades") or []
    if eq is not None and len(eq) >= 2:
        btx = float(eq.iloc[-1] / eq.iloc[0])
        st.metric("BT× (Strategy factor)", f"{btx:.4f}")
    if bh is not None and len(bh) >= 1:
        st.metric("BH× (Buy&Hold factor)", f"{float(bh.iloc[-1]):.4f}")
    st.metric("Trades", f"{len(tr):d}")
