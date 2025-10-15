# -*- coding: utf-8 -*-
from __future__ import annotations

import json, time, random, pathlib
from datetime import date, timedelta
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Branding (frivillig)
try:
    from app.branding import apply as brand
except Exception:
    def brand(*a, **k): pass

brand(page_title="Dala Trader – Optimizer", page_icon="⚙️")
st.title("⚙️ Optimizer")
st.caption("Optimerar tre strategier per ticker över valt intervall. Hämtar OHLCV en gång och kör allt lokalt för fart.")

# ---- Imports för motorerna
from app.data_providers import get_ohlcv as GET_OHLCV
from app.btwrap import run_backtest as RUN_BT

# ---- Hjälp
def ur(rng: random.Random, a: float, b: float) -> float:
    return a + (b - a) * rng.random()

def make_params(rng: random.Random) -> Dict[str, Any]:
    use_trend_filter = bool(rng.getrandbits(1))
    use_macd_filter  = bool(rng.getrandbits(1))
    use_bb_filter    = bool(rng.getrandbits(1))
    use_stop_loss    = bool(rng.getrandbits(1))
    use_atr_trailing = bool(rng.getrandbits(1))
    trend_ma_type    = rng.choice(["SMA","EMA"])
    return {
        "use_rsi_filter": True,
        "rsi_window": rng.randint(8, 32),
        "rsi_min": ur(rng, 5.0, 35.0),
        "rsi_max": ur(rng, 60.0, 85.0),

        "use_trend_filter": use_trend_filter,
        "trend_ma_type": trend_ma_type,
        "trend_ma_window": rng.randint(20, 200),

        "breakout_lookback": rng.randint(20, 120),
        "exit_lookback":     rng.randint(10, 60),

        "use_macd_filter": use_macd_filter,
        "macd_fast":   rng.randint(8, 16),
        "macd_slow":   rng.randint(18, 30),
        "macd_signal": rng.randint(8, 14),

        "use_bb_filter": use_bb_filter,
        "bb_window": rng.randint(15, 30),
        "bb_nstd":   ur(rng, 1.6, 2.4),
        "bb_min":    ur(rng, 0.0, 0.8),

        "use_stop_loss": use_stop_loss,
        "stop_mode": rng.choice(["pct","atr"]),
        "stop_loss_pct": ur(rng, 0.03, 0.20),

        "atr_window": rng.randint(10, 20),
        "atr_mult":   ur(rng, 1.2, 3.2),

        "use_atr_trailing": use_atr_trailing,
        "atr_trail_mult":   ur(rng, 1.2, 3.5),
    }

def score(metrics: Dict[str, Any]) -> float:
    tr = float(metrics.get("TotalReturn") or 0.0)
    sh = float(metrics.get("SharpeD") or 0.0)
    mdd = float(metrics.get("MaxDD") or 0.0)  # negativt tal
    return 2.0*tr + 1.0*sh + 0.5*(-mdd)

def five_year_window():
    end = date.today()
    start = end - timedelta(days=365*5 + 2)
    return start, end

@st.cache_data(show_spinner=False)
def load_ohlcv_cached(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = GET_OHLCV(ticker=ticker, start=start, end=end)
    return df

def run_optimizer_ui(ticker: str, sims: int, seed: int, start: str, end: str):
    t0 = time.time()
    df = load_ohlcv_cached(ticker, start, end)
    if df is None or len(getattr(df, "index", [])) == 0:
        st.error("Tomt OHLCV för perioden.")
        return

    rng = random.Random(seed)
    best: List[Tuple[float, Dict[str,Any], Dict[str,Any]]] = []
    prog = st.progress(0)
    status = st.empty()

    for i in range(1, sims+1):
        p = make_params(rng)
        p["from_date"] = start
        p["to_date"]   = end
        try:
            res = RUN_BT(df, p)
            m = res.get("summary", {}) if isinstance(res, dict) else {}
            s = score(m)
            best.append((s, p, m))
            if len(best) > 16:
                best.sort(key=lambda x: x[0], reverse=True)
                best = best[:16]
        except Exception as e:
            # svälj enstaka fel
            pass

        if (i % max(1, sims//100)) == 0 or i == sims:
            prog.progress(int(i*100/sims))
            status.text(f"Kör sim {i}/{sims} …")

    best.sort(key=lambda x: x[0], reverse=True)
    top = best[:3]
    names = ["conservative", "balanced", "aggressive"]
    profiles = []
    for idx, (s, p, m) in enumerate(top):
        profiles.append({
            "name": f"{ticker} – {names[idx]}",
            "ticker": ticker,
            "params": p,
            "metrics": m,
        })

    outdir = pathlib.Path("profiles")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{ticker}.json"
    outfile.write_text(json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2), encoding="utf-8")

    st.success(f"Klar. Sparade {len(profiles)} profiler → {outfile}")
    # visa tabell
    rows = []
    for pr in profiles:
        m = pr.get("metrics", {})
        rows.append({
            "Name": pr.get("name",""),
            "TotalReturn": m.get("TotalReturn"),
            "SharpeD": m.get("SharpeD"),
            "MaxDD": m.get("MaxDD"),
            "Trades": m.get("Trades"),
            "FinalEquity": m.get("FinalEquity"),
            "BuyHold": m.get("BuyHold"),
        })
    st.dataframe(pd.DataFrame(rows))

    # applicera bästa till sessionen
    best_params = profiles[0]["params"] if profiles else None
    if best_params and st.button(f"✅ Använd bästa för {ticker} i Backtest"):
        for k, v in best_params.items():
            st.session_state[k] = v
        st.session_state["ticker"] = ticker
        st.success("Applicerat till session. Gå till Backtest och kör.")

# ---------- UI-kontroller ----------
col1, col2, col3, col4 = st.columns([1.1,1,1,1])
with col1:
    ticker = st.text_input("Ticker", value=st.session_state.get("ticker","GETI B")).strip()
with col2:
    sims = st.number_input("Antal simuleringar", min_value=100, max_value=100_000, value=1000, step=100)
with col3:
    seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)
with col4:
    last5 = st.checkbox("5 år bakåt (auto)", value=True)

if last5:
    s, e = five_year_window()
    start, end = s.isoformat(), e.isoformat()
else:
    s = st.date_input("Start", value=five_year_window()[0])
    e = st.date_input("Slut", value=five_year_window()[1])
    start, end = s.isoformat(), e.isoformat()

st.markdown("---")
if st.button("🚀 Kör optimering"):
    if not ticker:
        st.error("Ange en ticker.")
    else:
        run_optimizer_ui(ticker, int(sims), int(seed), start, end)
else:
    st.info("Tips: börja med 1 000–2 000 simuleringar. Öka när det ser rimligt ut.")
