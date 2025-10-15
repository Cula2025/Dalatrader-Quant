from __future__ import annotations
import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from app.data_providers import get_ohlcv
from app.backtest import run_backtest  # används som tidigare


# ----------------------------- Hjälp: data -----------------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(df).copy()
    if "Date" not in x.columns:
        idx = x.index.name or "Date"
        x = x.reset_index().rename(columns={idx: "Date"})
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
    for c in ("Open", "High", "Low", "Close", "Volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return x


# ----------------------------- Stödda nycklar -----------------------------
# Backtrackern filtrerar inkommande kwargs mot ALLOWED för att undvika
# TypeError i backtestern om den inte stödjer någon nyckel.
ALLOWED: set[str] = {
    # RSI + trend
    "strategy",
    "use_rsi_filter", "rsi_window", "rsi_min", "rsi_max",
    "use_trend_filter", "trend_ma_type", "trend_ma_window",

    # Breakout / Exit
    "breakout_lookback", "exit_lookback",

    # MACD
    "use_macd_filter", "macd_fast", "macd_slow", "macd_signal",

    # Bollinger %B
    "use_bb_filter", "bb_window", "bb_nstd", "bb_min",

    # Stop / ATR (både fixed och trailing – ok om backtest ignorerar det ena)
    "use_stop_loss", "stop_mode", "stop_loss_pct",
    "atr_window", "atr_mult",
    "use_atr_trail", "atr_trail_mult",
}

# Säkra defaults om profil saknar viss nyckel
DEFAULTS: Dict[str, Any] = {
    "strategy": "rsi",

    "use_rsi_filter": True,
    "rsi_window": 14,
    "rsi_min": 25.0,
    "rsi_max": 60.0,

    "use_trend_filter": False,
    "trend_ma_type": "EMA",
    "trend_ma_window": 100,

    "breakout_lookback": 55,
    "exit_lookback": 20,

    "use_macd_filter": False,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    "use_bb_filter": False,
    "bb_window": 20,
    "bb_nstd": 2.0,
    "bb_min": 0.2,

    "use_stop_loss": False,
    "stop_mode": "pct",      # "pct" | "atr"
    "stop_loss_pct": 0.08,
    "atr_window": 14,
    "atr_mult": 2.0,

    "use_atr_trail": False,
    "atr_trail_mult": 2.0,
}


def clamp_params(p: Dict[str, Any]) -> Dict[str, Any]:
    """Fusera profil med defaults och filtrera ner till ALLOWED."""
    base = dict(DEFAULTS)
    base.update(p or {})
    return {k: base[k] for k in base.keys() if k in ALLOWED}


# ----------------------------- Plott -----------------------------
def plot_equity_with_trades(eq_df: pd.DataFrame,
                            trades: pd.DataFrame,
                            title: str = "") -> alt.Chart:
    if eq_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()

    base = alt.Chart(eq_df.rename(columns={"Date": "date"})).properties(height=320)
    equity_line = base.mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Equity:Q", title="Equity")
    )

    pts = pd.DataFrame()
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        t_buy = trades[["EntryTime", "EntryPrice"]].rename(
            columns={"EntryTime": "date", "EntryPrice": "price"})
        t_buy["kind"] = "Buy"

        t_sell = trades[["ExitTime", "ExitPrice"]].rename(
            columns={"ExitTime": "date", "ExitPrice": "price"})
        t_sell["kind"] = "Sell"

        pts = pd.concat([t_buy, t_sell], ignore_index=True)
        pts = pts.dropna()

    if not pts.empty:
        pts["date"] = pd.to_datetime(pts["date"], errors="coerce")
        marks = alt.Chart(pts).mark_point(size=60, filled=True).encode(
            x="date:T",
            y=alt.Y("price:Q", title=""),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["Buy", "Sell"],
                                                      range=["#2ca02c", "#d62728"]))
        )
        return (equity_line + marks).properties(title=title)
    else:
        return equity_line.properties(title=title)


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Dalatrader – Backtracker", layout="wide")
st.title("Backtracker (profiler från Optimizer)")

# Datum, ticker
today = dt.date.today()
col0, col1, col2 = st.columns([2, 1, 1])
with col0:
    ticker = st.text_input("Ticker", value="VOLV B")
with col1:
    start = st.text_input("Start (YYYY/MM/DD)",
                          value=(today - dt.timedelta(days=365 * 5)).strftime("%Y/%m/%d"))
with col2:
    end = st.text_input("Slut (YYYY/MM/DD)", value=today.strftime("%Y/%m/%d"))

# Profilinmatning
st.subheader("Profiler")
cA, cB = st.columns([1.5, 2.5])
with cA:
    up = st.file_uploader("Ladda upp JSON-fil (från Optimizer)", type=["json"])
with cB:
    pasted = st.text_area("Klistra in JSON (valfritt) – t.ex. output från Optimizer",
                          height=120,
                          placeholder='{"profiles":[{"name":"VOLV B – balanced","ticker":"VOLV B","params":{...}}, ...]}')

state = st.session_state
state.setdefault("profiles", [])
state.setdefault("df", pd.DataFrame())

def load_profiles_from_payload(txt: str) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(txt)
    except Exception:
        return []
    profs = payload.get("profiles") or []
    out = []
    for pr in profs:
        nm = pr.get("name") or pr.get("label") or "Profile"
        params = pr.get("params") or {}
        out.append({"name": nm, "params": clamp_params(params)})
    return out

# Läs upp/paste
if up is not None:
    try:
        prof_text = up.read().decode("utf-8")
        state["profiles"] = load_profiles_from_payload(prof_text)
        st.success(f"Läste {len(state['profiles'])} profilers paramset från fil.")
    except Exception as e:
        st.error(f"Kunde inte läsa fil: {e}")

if pasted.strip():
    profs = load_profiles_from_payload(pasted)
    if profs:
        state["profiles"] = profs
        st.success(f"Läste {len(state['profiles'])} profilers paramset från text.")

# Lista val
names = [p["name"] for p in state["profiles"]]
chosen = st.selectbox("Välj profil att ladda", options=["(ingen)"] + names, index=0)

# Visa params (read-only) och knappar
sel_params: Dict[str, Any] = {}
if chosen != "(ingen)":
    idx = names.index(chosen)
    sel_params = state["profiles"][idx]["params"]
    with st.expander("Parametrar i vald profil", expanded=False):
        st.json(sel_params)

col_btn_fetch, col_btn_run = st.columns(2)
with col_btn_fetch:
    if st.button("Hämta data", type="primary"):
        try:
            df = get_ohlcv(ticker=ticker, start=start, end=end, source="borsdata")
            state["df"] = normalize_df(df)
            st.success(f"Läste {len(state['df'])} rader. Period: "
                       f"{state['df']['Date'].iloc[0].date()} → {state['df']['Date'].iloc[-1].date()}")
            st.dataframe(state["df"].tail(5), width='stretch')
        except Exception as e:
            st.error(f"Kunde inte ladda/städa data: {e}")

with col_btn_run:
    run_all = st.button("Kör backtest på vald profil")

# Kör
if run_all:
    df = state.get("df", pd.DataFrame())
    if df.empty:
        st.warning("Hämta data först.")
    elif not sel_params:
        st.warning("Välj en profil ovan.")
    else:
        # Sista skydd: filtrera ner params
        kwargs = clamp_params(sel_params)

        try:
            res = run_backtest(df.copy(), **kwargs)
        except TypeError:
            # Om din backtester inte tar kwargs utan en dict kan vi försöka som **kwargs igen
            # men utan nycklar som orsakade fel. Här gör vi inget mer – antag att ALLOWED räcker.
            raise

        summ = res.get("summary", {}) or {}
        eq   = res.get("equity") or res.get("equity_buy") or pd.DataFrame()
        if isinstance(eq, pd.DataFrame) and "Date" not in eq.columns and not eq.empty:
            # om backtestern returnerar med index som datum
            eq = eq.reset_index().rename(columns={"index": "Date"})

        trades = res.get("trades", pd.DataFrame())
        if isinstance(trades, list):
            trades = pd.DataFrame(trades)

        # KPI-panel
        m1, m2, m3, m4, m5 = st.columns(5)
        def pct(x): 
            try: 
                return f"{float(x)*100:.2f}%"
            except: 
                return "0.00%"
        m1.metric("Total return", pct(summ.get("TotalReturn", 0.0)))
        m2.metric("Buy & Hold",   pct(summ.get("BuyHold", 0.0)))
        m3.metric("MaxDD",        pct(summ.get("MaxDD", 0.0)))
        m4.metric("CAGR",         pct(summ.get("CAGR", 0.0)))
        m5.metric("Trades",       f"{int(summ.get('Trades', 0))}")

        # Graf + markeringar
        chart = plot_equity_with_trades(eq, trades, title=chosen)
        st.altair_chart(chart, width='stretch')

        # Trades tabell
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            keep = [c for c in ["EntryTime","EntryPrice","ExitTime","ExitPrice","PnL","reason"] if c in trades.columns]
            st.dataframe(trades[keep].sort_values("EntryTime", ascending=False),
                         width='stretch')
        else:
            st.caption("Inga trades att visa.")

        with st.expander("Rå summering"):
            st.json(summ)
