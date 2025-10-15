import streamlit as st
import pandas as pd
import json
from pathlib import Path

from app.portfolio_math import pick_first, equal_weight_rebalanced
from app.equity_extract import extract_equity
from app.data_providers import get_ohlcv

try:
    from app.backtracker import run_backtest as RUN_BT
    ENGINE = "backtracker"
except Exception:
    from app.btwrap import run_backtest as RUN_BT
    ENGINE = "btwrap"

st.title("Portfolio V2 – EW (test)")
st.caption(f"Motor: {ENGINE}  ·  Rebalansering: daglig lika vikt")

def load_profiles():
    rows=[]
    for p in sorted(Path("profiles").glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for pr in data.get("profiles", []):
                t = pr.get("ticker") or (pr.get("params") or {}).get("ticker")
                if not t: 
                    continue
                rows.append({
                    "ticker": t,
                    "name": pr.get("name"),
                    "params": pr.get("params") or {},
                    "file": p.name,
                    "metrics": pr.get("metrics") or {},
                })
        except Exception:
            pass
    return rows

rows = load_profiles()

# Välj bästa profil per ticker (högst TotalReturn)
best_by_ticker = {}
for r in rows:
    tr = float(r["metrics"].get("TotalReturn", float("nan")))
    if r["ticker"] not in best_by_ticker or tr > float(best_by_ticker[r["ticker"]]["metrics"].get("TotalReturn", float("-inf"))):
        best_by_ticker[r["ticker"]] = r

tickers = sorted(best_by_ticker.keys())
sel_tickers = st.multiselect("Universum (bästa profil/ticker)", tickers, default=tickers)

index_ticker = st.text_input("Jämförelseindex (valfritt, t.ex. OMX30 – om stöds)", "OMX30")

selected_profiles = [best_by_ticker[t] for t in sel_tickers]

# Kör profiler → hämta equity-serier
equities = []
with st.spinner("Kör backtester..."):
    for pr in selected_profiles:
        res = RUN_BT(p={"ticker": pr["ticker"], "params": pr["params"]})
        x = pick_first(res.get("equity"), res.get("summary"), res)
        s = extract_equity(x)
        equities.append(s)

port = equal_weight_rebalanced(equities)

def safe_bh_curve(ticker: str, start: str):
    try:
        df = get_ohlcv(ticker=ticker, start=start, end=None)
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if s.empty:
            return None
        s = s / float(s.iloc[0])
        s.name = ticker
        return s
    except Exception:
        return None

index_curve = None
if len(port):
    index_curve = safe_bh_curve(index_ticker, port.index[0].strftime("%Y-%m-%d"))

# Plot
import altair as alt
if len(port):
    df = pd.DataFrame({"Portfolio": port})
    if index_curve is not None:
        ic = index_curve.reindex(port.index).dropna()
        df = df.join(ic, how="inner")
    chart = alt.Chart(df.reset_index().rename(columns={"index":"Date"})).mark_line(point=False, interpolate='monotone', strokeWidth=2.5).encode(
        x='Date:T'
    )
    layers = chart.encode(y='Portfolio:Q', color=alt.value('#1f77b4'))
    if index_curve is not None:
        layers = layers + chart.encode(y=f'{index_ticker}:Q', color=alt.value('#9aa0a6'))
    st.altair_chart(layers, use_container_width=True)
    st.metric("Slutvärde (Portfolio)", f"{float(port.iloc[-1]):.4f}×")
else:
    st.info("Ingen portföljkurva ännu – välj tickers ovan.")

