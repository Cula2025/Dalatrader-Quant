# 📈 Portfolio V2 – linjegraf (Strategi, Buy&Hold, Index)
# Felsäker: en robust to_step, en enda BH-beräkning, tydlig ordning.

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import streamlit as st

from app import btwrap as W
from app.equity_extract import extract_equity
from app.portfolio_math import equal_weight_rebalanced
from app.data_providers import get_ohlcv

st.set_page_config(page_title="📈 Portfolio V2 – linjegraf", layout="wide")
st.title("📈 Portfolio V2 – linjegraf (Strategi, Buy&Hold, Index)")

INDEX_TICKER = "OMXS30GI"  # GI-index

# --- Tillfälligt: släck grafer om vi vill isolera logiken (låt gärna vara på nu) ---
# st.caption("🔧 Chart-rendering kan slås av här om vi vill felsöka utan grafer.")

# -----------------------
# Hjälpfunktioner (överst)
# -----------------------
def pick_first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def pick_best_profile(d: Dict[str, Any]) -> Dict[str, Any] | None:
    profs = d.get("profiles") or []
    if not profs:
        return None
    def tr(p):
        m = p.get("metrics") or {}
        v = m.get("TotalReturn")
        try: return float(v) if v is not None else float("-inf")
        except: return float("-inf")
    return sorted(profs, key=tr, reverse=True)[0]

def to_equity_series(ticker: str, params: Dict[str, Any]) -> pd.Series:
    res = W.run_backtest(p={"ticker": ticker, "params": dict(params or {})})
    x   = pick_first(res.get("equity", None), res.get("summary", None), res)
    s   = extract_equity(x)
    s   = pd.to_numeric(s, errors="coerce").dropna()
    return s

def buyhold_equal_weight(tickers: List[str], start: str | None) -> pd.Series:
    """EW buy&hold över tickers: köp dag1, ingen rebal, normaliserad till 1.0."""
    if not tickers:
        return pd.Series(dtype="float64")
    prices = []
    for t in tickers:
        df = get_ohlcv(ticker=t, start=start, end=None)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if s.empty:
            continue
        s = s / float(s.iloc[0])
        prices.append(s)
    if not prices:
        return pd.Series(dtype="float64")
    P = pd.concat(prices, axis=1, join="inner")
    P.columns = [f"bh_{i+1}" for i in range(P.shape[1])]
    bh = P.mean(axis=1)
    bh.name = "Buy&Hold"
    return bh

def index_curve(index_ticker: str, start: str | None) -> pd.Series:
    df = get_ohlcv(index_ticker, start=start, end=None)
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype="float64")
    s = s / float(s.iloc[0])
    s.name = index_ticker
    return s

def to_step(s, name: str) -> pd.Series:
    """Robust – accepterar None, scalar, list, Series, DataFrame(first col)."""
    import numpy as np
    if s is None or isinstance(s, (float, int, np.floating, np.integer)):
        return pd.Series(dtype="float64", name=name)
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return pd.Series(dtype="float64", name=name)
        s = s.iloc[:, 0]
    if not isinstance(s, pd.Series):
        try:
            s = pd.Series(s)
        except Exception:
            return pd.Series(dtype="float64", name=name)
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = name
    if s.empty:
        return s
    return s.reset_index(drop=True)

def lvl(s: pd.Series | None) -> float | None:
    try: return float(s.iloc[-1])
    except Exception: return None

# -----------------------
# UI: välj profiler
# -----------------------
all_files = sorted([str(p) for p in Path("profiles").glob("*.json")])
sel_files = st.multiselect("Profilfiler (profiles/*.json)", all_files, default=all_files[:5])
if not sel_files:
    st.info("Välj minst en profilfil.")
    st.stop()

# -----------------------
# Läs profiler → equity-serier
# -----------------------
records: List[Dict[str, Any]] = []
tickers: List[str] = []
from_dates: List[str] = []

for f in sel_files:
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        pr = pick_best_profile(d)
        if not pr:
            continue
        t = pr.get("ticker") or (pr.get("params") or {}).get("ticker")
        params = dict(pr.get("params") or {})
        if not t:
            continue
        eq = to_equity_series(t, params)
        if eq is None or eq.empty:
            continue
        records.append({"file": f, "ticker": t, "params": params, "equity": eq})
        tickers.append(t)
        fd = params.get("from_date")
        if fd: from_dates.append(str(fd))
    except Exception as e:
        st.warning(f"Misslyckades läsa {f}: {type(e).__name__}: {e}")

if not records:
    st.error("Inga equity-kurvor kunde läsas.")
    st.stop()

# -----------------------
# Portfölj (rebalance), BH & Index
# -----------------------
rb = equal_weight_rebalanced([r["equity"] for r in records])
rb = pd.to_numeric(rb, errors="coerce").dropna()
rb = rb / float(rb.iloc[0])  # normalisera

start = min(from_dates) if from_dates else None
bh   = buyhold_equal_weight(tickers, start)
idx  = index_curve(INDEX_TICKER, start)

# -----------------------
# Align & graf/nyckeltal
# -----------------------
rb_s  = to_step(rb,  "Portfolio")
bh_s  = to_step(bh,  "Buy&Hold")
idx_s = to_step(idx, INDEX_TICKER)

n = min([len(x) for x in [rb_s, bh_s, idx_s] if x is not None and len(x) > 0] or [0])
if n == 0:
    st.error("Kunde inte aligna kurvorna (saknar data).")
    st.stop()

rb_s  = rb_s.iloc[:n]
bh_s  = bh_s.iloc[:n]
idx_s = idx_s.iloc[:n]

df_plot = pd.DataFrame({
    "step": range(n),
    "Portfolio": rb_s.values,
    "Buy&Hold":  bh_s.values,
    INDEX_TICKER: idx_s.values,
}).set_index("step")

st.line_chart(df_plot)

m_port = lvl(rb_s); m_bh = lvl(bh_s); m_idx = lvl(idx_s)
c1, c2, c3 = st.columns(3)
c1.metric("Portfolio", f"{m_port:.4f}×" if m_port is not None else "—")
c2.metric("Buy&Hold",  f"{m_bh:.4f}×"   if m_bh   is not None else "—")
c3.metric(INDEX_TICKER, f"{m_idx:.4f}×" if m_idx  is not None else "—")

# Sanity: BH per ticker
with st.expander("Visa Buy&Hold per ticker (sanity)"):
    rows = []
    for t in tickers:
        df = get_ohlcv(t, start=start, end=None)
        if df is None or df.empty:
            continue
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({"ticker": t, "BH_ratio": float(s.iloc[-1] / s.iloc[0])})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("ticker"))
