import json
from pathlib import Path
import pandas as pd
import streamlit as st

# Motor + hjälpare
try:
    from app.backtracker import run_backtest
except Exception:
    from app import btwrap as W
    run_backtest=W.run_backtest

from app.equity_extract import extract_equity
from app.data_providers import get_ohlcv

st.set_page_config(page_title="Portfolio – Auto Universe", layout="wide")
st.title("Portfolio – Auto Universe (bäst per aktie)")

def first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def bh_curve(ticker: str, start: str|None):
    tk = "OMXS30GI" if (ticker or "").upper() in ("OMX30","^OMX30") else ticker
    df = get_ohlcv(ticker=tk, start=start, end=None)
    if df is None or df.empty or "Close" not in df.columns: 
        return pd.Series(dtype="float64")
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if s.empty: return pd.Series(dtype="float64")
    return (s / s.iloc[0]).rename(tk)

# 2.1 Läs auto-universum
auto_path = Path("profiles/_auto_universe.json")
if not auto_path.exists():
    st.error("Hittar inte profiles/_auto_universe.json – kör verktyget på nytt.")
    st.stop()

data = json.loads(auto_path.read_text(encoding="utf-8"))
sel  = data.get("profiles") or []
if not sel:
    st.warning("Auto-universumet är tomt.")
    st.stop()

# 2.2 Visa universumlistan
df_uni = pd.DataFrame([{
    "Ticker": p.get("ticker"),
    "Profil": p.get("name"),
    "TR": (p.get("metrics") or {}).get("TotalReturn"),
    "From": (p.get("params") or {}).get("from_date"),
    "Källa": p.get("_source"),
} for p in sel])
st.subheader("Universum (bästa per fil)")
st.dataframe(df_uni, hide_index=True, use_container_width=True)

# 2.3 Kör alla samtidigt (lika vikt, daglig rebalansering)
common_start = max((p.get("params") or {}).get("from_date") for p in sel)
st.caption(f"Gemensam start: {common_start}")

eq_curves = {}
trade_counts = []
for p in sel:
    t = p.get("ticker") or (p.get("params") or {}).get("ticker")
    params = dict(p.get("params") or {})
    res = run_backtest(p={"ticker": t, "params": params})

    x = first_not_none(res.get("equity"), res.get("summary"), res)
    eq = extract_equity(x)
    eq = pd.to_numeric(eq, errors="coerce").dropna()
    eq_curves[t] = eq.rename(t)

    tr = res.get("trades")
    ntr = (len(tr) if tr is not None else 0)
    trade_counts.append({"Ticker": t, "Trades": ntr})

# 2.4 Bygg portföljkurva (inner join på datum + lika vikt)
E = pd.concat(eq_curves.values(), axis=1, join="inner")
E = E.loc[E.index >= pd.to_datetime(common_start)]
portfolio = E.mean(axis=1).rename("Portfolio")

# 2.5 Indexjämförelse
index_ticker = st.selectbox("Jämför mot index", ["OMXS30GI","OMXS30","(ingen)"], index=0)
idx = pd.Series(dtype="float64")
if index_ticker != "(ingen)":
    idx = bh_curve(index_ticker, common_start)
    idx = idx.reindex(portfolio.index, method="ffill").dropna()
    idx.name = index_ticker

# 2.6 Plotta
chart_df = pd.DataFrame({"Portfolio": portfolio})
for t, s in eq_curves.items():
    chart_df[t] = s.reindex(portfolio.index, method="ffill")
if not idx.empty:
    chart_df[idx.name] = idx

st.subheader("Kurvor (normaliserade, 1.0 vid start)")
st.line_chart(chart_df, use_container_width=True)

# 2.7 Snabb sammanfattning + trades
st.subheader("Snabbdata")
col1, col2, col3 = st.columns(3)
col1.metric("Universe", len(sel))
col2.metric("Startvärde", f"{float(portfolio.iloc[0]):.2f}×")
col3.metric("Slutvärde",  f"{float(portfolio.iloc[-1]):.2f}×")

st.subheader("Antal trades (per aktie)")
st.dataframe(pd.DataFrame(trade_counts), hide_index=True, use_container_width=True)
