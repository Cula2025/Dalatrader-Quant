import json, pathlib
import pandas as pd
import streamlit as st
from typing import Dict, Any

# Motor: försök backtracker först, annars btwrap
try:
    from app.backtracker import run_backtest
    ENGINE = "backtracker"
except Exception:
    from app import btwrap as W
    run_backtest = W.run_backtest  # type: ignore
    ENGINE = "btwrap"

from app.data_providers import get_ohlcv
from app.equity_extract import extract_equity

st.set_page_config(page_title="Portfolio V2 (tabs)", page_icon="📚", layout="wide")
st.title("📚 Portfolio V2 (tabs)")
st.caption(f"Engine: **{ENGINE}** · Likaviktad portfölj över valda profiler. Robust mot tomma DF/None.")

def pick_equity_payload(res: Dict[str, Any]):
    # Undvik bool-cast på DataFrame
    for k in ("equity","summary"):
        if k in res and res[k] is not None:
            return res[k]
    return res

def safe_bh_curve(ticker: str, start: str|None):
    df = get_ohlcv(ticker=(("OMXS30GI" if (ticker or "").upper() in ("OMX30","^OMX30") else ticker)), start=start, end=None)
    if df is None or len(df) < 2 or "Close" not in df.columns:
        return None
    s = (df["Close"] / float(df["Close"].iloc[0])).astype("float64")
    s.name = f"BH:{ticker}"
    return s

def align_mean(curves: Dict[str, pd.Series]) -> pd.Series|None:
    if not curves:
        return None
    df = pd.concat(curves.values(), axis=1)
    # framåt-fyll luckor och droppa rader där någon fortfarande saknas
    df = df.ffill().dropna()
    if df.empty:
        return None
    m = df.mean(axis=1)
    # rebase till 1.0
    m = m / float(m.iloc[0])
    m.name = "Portfolio"
    return m

# --- Sidopanel / val ---
proffiles = sorted(pathlib.Path("profiles").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not proffiles:
    st.warning("Hittar inga profilfiler i profiles/*.json")
    st.stop()

left, right = st.columns([1,2])
with left:
    fname = st.selectbox("Profilfil", [p.name for p in proffiles])
    index_ticker = st.text_input("Jämförelseindex-ticker (Börsdata)", value="OMX30")
data = json.loads((pathlib.Path("profiles")/fname).read_text(encoding="utf-8"))
plist = data.get("profiles") or []
if not plist:
    st.warning("Inga profiler i filen.")
    st.stop()

names = [p.get("name","<namn saknas>") for p in plist]
chosen = st.multiselect("Välj profiler (portföljens universum)", names, default=names)
sel = [p for p in plist if p.get("name","") in set(chosen)]
if not sel:
    st.info("Välj minst en profil.")
    st.stop()

# --- Kör backtest för valda profiler ---
eq_series: Dict[str, pd.Series] = {}
bh_series: Dict[str, pd.Series] = {}
rows = []

for p in sel:
    name   = p.get("name","<namn saknas>")
    params = dict(p.get("params") or {})
    ticker = p.get("ticker") or params.get("ticker")
    with st.spinner(f"Kör {name} / {ticker}…"):
        res = run_backtest(p={"ticker": ticker, "params": params})
    eq = extract_equity(pick_equity_payload(res))
    if len(eq):
        eq_series[name] = pd.Series(eq.values, index=eq.index, name=name)
        tr_calc = float(eq.iloc[-1]) - 1.0
        final_eq = float(eq.iloc[-1])
    else:
        tr_calc = float("nan")
        final_eq = float("nan")

    # BH per instrument
    bh = safe_bh_curve(ticker, params.get("from_date"))
    if bh is not None:
        bh_series[name] = bh.rename(f"BH:{name}")
        bh_ret = float(bh.iloc[-1] - 1.0)
    else:
        bh_ret = float("nan")

    met = res.get("metrics") or {}
    trades = met.get("Trades")
    if trades is None:
        trades = len(res.get("trades", []))
    maxdd = met.get("MaxDD")

    rows.append({
        "Name": name, "Ticker": ticker,
        "TR (calc)": tr_calc, "FinalEquity": final_eq,
        "BH (calc)": bh_ret,
        "Trades": trades if trades is not None else 0,
        "MaxDD": float(maxdd) if maxdd is not None else float("nan"),
    })

df_universe = pd.DataFrame(rows).set_index("Name")

# Portföljkurvor (likaviktad)
portfolio_eq = align_mean(eq_series)      # strategi
portfolio_bh = align_mean(bh_series)      # buy/hold

# Index-kurva
index_curve = safe_bh_curve(index_ticker, min((p.get("params") or {}).get("from_date") for p in sel))
if index_curve is not None and len(index_curve):
    index_curve = (index_curve / float(index_curve.iloc[0]))
    index_curve.name = index_ticker

# --- UI: Tabs ---
tab_uni, tab_trades, tab_cmp = st.tabs(["🔎 Universum", "📜 Trades", "📈 Jämförelse"])

with tab_uni:
    st.subheader("Universum (per profil)")
    st.dataframe(df_universe, use_container_width=True)

with tab_trades:
    st.subheader("Trades")
    total_trades = int(df_universe["Trades"].fillna(0).sum()) if "Trades" in df_universe else 0
    st.metric("Totalt antal trades", total_trades)
    # visa första befintliga trade-lista om motorn lämnar den
    for p in sel:
        name = p.get("name","")
        params = dict(p.get("params") or {})
        ticker = p.get("ticker") or params.get("ticker")
        # försök att få ut listan igen utan att köra om motorn
        # (hade vi sparat 'res' ovan skulle vi kunna visa full logik per profil)
        break  # enkel summering i detta steg

with tab_cmp:
    st.subheader("Jämförelsekurvor")
    curves = {}
    if portfolio_eq is not None: curves["Portfölj (STRAT)"] = portfolio_eq
    if portfolio_bh is not None: curves["Portfölj (BH)"]    = portfolio_bh
    if index_curve is not None:  curves[index_ticker]       = index_curve
    if curves:
        df_plot = pd.concat(curves.values(), axis=1).ffill().dropna()
        st.line_chart(df_plot)
        last = df_plot.iloc[-1]
        c1,c2,c3 = st.columns(3)
        if "Portfölj (STRAT)" in df_plot.columns:
            c1.metric("Portfölj STRAT", f"{float(last['Portfölj (STRAT)']-1.0):.4f}×")
        if "Portfölj (BH)" in df_plot.columns:
            c2.metric("Portfölj BH", f"{float(last['Portfölj (BH)']-1.0):.4f}×")
        if index_curve is not None and index_ticker in df_plot.columns:
            c3.metric(index_ticker, f"{float(last[index_ticker]-1.0):.4f}×")
    else:
        st.info("Inga kurvor att visa ännu.")
