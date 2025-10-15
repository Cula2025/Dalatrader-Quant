from __future__ import annotations
import traceback, datetime as dt
from pathlib import Path
import streamlit as st, pandas as pd

st.set_page_config(page_title="Smoke: INVE B (en profil)", page_icon="?", layout="wide")

PRIMARY = "#1f6feb"
st.markdown(f"<style>.block-container{{padding-top:.75rem}} .stButton>button{{background:{PRIMARY};color:#fff;border:0}}</style>", unsafe_allow_html=True)

def _fail(msg, exc=None):
    st.error(msg)
    if exc:
        with st.expander("Teknisk detalj"): st.code(traceback.format_exc())
    st.stop()

# ---- Fast profil (din JSON) ----
PROFILE = {
    "name": "INVE B � auto_best_1000",
    "ticker": "INVE B",
    "params": {
        "strategy":"rsi","trend_ma_type":"EMA","use_trend_filter":False,"trend_ma_window":0,
        "fast":15,"slow":100,
        "use_rsi_filter":True,"rsi_window":7,"rsi_min":25.0,"rsi_max":75.0,
        "breakout_lookback":55,"exit_lookback":20,
        "use_macd_filter":False,"macd_fast":12,"macd_slow":26,"macd_signal":9,"macd_mode":"above_zero",
        "use_bb_filter":False,"bb_window":20,"bb_nstd":2.0,"bb_mode":"exit_below_mid","bb_percent_b_min":0.8,
        "atr_window":14,"atr_stop_mult":0.0,"atr_trail_mult":0.0,
        "cost_bps":0.0,"cash_rate_apy":0.0,"max_positions":1,"per_trade_pct":100.0,"max_exposure_pct":100.0
    },
    "metrics": {
        "TotalReturn": 2.499694107703308,
        "MaxDD": -0.25822302796110674,
        "SharpeD": 1.1606823192991103,
        "BuyHold": 1.191204588910134
    }
}

# ---- UI: period + k�lla ----
st.header("INVE B � smoke test (en profil)")
source = st.selectbox("Datak�lla", ["borsdata","stooq","yahoo"], index=0)
today = dt.date.today()
col1,col2 = st.columns(2)
with col1: start_date = st.date_input("Fr�n", value=today - dt.timedelta(days=365*5))
with col2: end_date   = st.date_input("Till", value=today)

run = st.button("?? K�r INVE B", width='stretch')
if not run: st.stop()

# ---- Data ----
try:
    from app.data_providers import get_ohlcv
    raw = get_ohlcv(PROFILE["ticker"], start=start_date.isoformat(), end=end_date.isoformat(), source=source)
except Exception as e:
    _fail(f"? Kunde inte h�mta data f�r {PROFILE['ticker']} ({source})", e)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return pd.DataFrame()
    if "Date" not in df.columns:
        idx = df.index.name or "Date"
        df = df.reset_index().rename(columns={idx:"Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ("Open","High","Low","Close","Volume"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

df = normalize_df(raw)
if df.empty: _fail("? Ingen data kunde laddas.")

# ---- K�r backtest ----
from app.backtest import run_backtest
SUPPORTED = {
    "strategy","trend_ma_type","use_trend_filter","trend_ma_window",
    "fast","slow",
    "use_rsi_filter","rsi_window","rsi_min","rsi_max",
    "breakout_lookback","exit_lookback",
    "use_macd_filter","macd_fast","macd_slow","macd_signal","macd_mode",
    "use_bb_filter","bb_window","bb_nstd","bb_mode","bb_percent_b_min",
    "atr_window","atr_stop_mult","atr_trail_mult",
    "cost_bps","cash_rate_apy","max_positions","per_trade_pct","max_exposure_pct",
    "slip_bps",
}
kwargs = {k:v for k,v in (PROFILE["params"] or {}).items() if k in SUPPORTED}

try:
    res = run_backtest(df, **kwargs)
except Exception as e:
    _fail(f"? Backtest misslyckades: {e}", e)

summary = res.get("summary", {}) or {}
equity_buy  = res.get("equity_buy")
equity_keep = res.get("equity_keep")
trades = res.get("trades")

# ---- Procentsiffror robust ----
def _pct(x):
    if x is None: return None
    try: x=float(x)
    except: return None
    return x*100 if -5.0 <= x <= 5.0 else x

# Strategi: f�rst fr�n summary, annars fr�n din JSON
strat_pct = _pct(summary.get("TotalReturn"))
if strat_pct is None and "TotalReturnPct" in summary:
    strat_pct = _pct(summary.get("TotalReturnPct"))
if strat_pct is None:
    strat_pct = _pct(PROFILE["metrics"].get("TotalReturn"))

# Buy&Hold ber�knad + din JSON som referens
def _bh(df):
    try:
        c0=float(df["Close"].iloc[0]); c1=float(df["Close"].iloc[-1])
        return (c1/c0 - 1)*100.0
    except: return None

bh_calc = _bh(df)
bh_json = _pct(PROFILE["metrics"].get("BuyHold"))

mdd = summary.get("MaxDD", PROFILE["metrics"].get("MaxDD"))
sh  = summary.get("SharpeD", PROFILE["metrics"].get("SharpeD"))
trades_n = int(summary.get("Trades", len(trades) if isinstance(trades, pd.DataFrame) else 0))

# ---- UI ----
st.subheader("Resultat")
c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("Total avkastning (strategi)", f"{strat_pct:.2f}%" if strat_pct is not None else "�")
with c2: st.metric("Buy & Hold (ber�knad)", f"{bh_calc:.2f}%" if bh_calc is not None else "�")
with c3: st.metric("Max Drawdown", f"{_pct(mdd):.2f}%" if mdd is not None else "�")
with c4: st.metric("Sharpe (daglig)", f"{float(sh):.2f}" if sh is not None else "�")
if bh_json is not None: st.caption(f"(BuyHold fr�n JSON): {bh_json:.2f}%")

st.subheader("Utveckling")
if isinstance(equity_buy, pd.DataFrame) and isinstance(equity_keep, pd.DataFrame):
    show = pd.DataFrame({
        "Buy (strategi)": equity_buy["Equity"].astype(float),
        "Keep (buy & hold)": equity_keep["Equity"].astype(float),
    })
    st.line_chart(show)
else:
    st.caption("Equity-kurvor saknas/ok�nda.")

st.subheader("Trades")
if isinstance(trades, pd.DataFrame) and not trades.empty:
    cols = [c for c in ["EntryTime","EntryPrice","ExitTime","ExitPrice","PnL","reason"] if c in trades.columns]
    st.dataframe(trades[cols].sort_values("ExitTime", ascending=False), width='stretch')
else:
    st.caption("Inga aff�rer i perioden eller tabell saknas.")

with st.expander("Debug"):
    st.write({"K�lla":source, "Period":f"{start_date}?{end_date}", "Bars":len(df)})
    st.write("Summary keys:", list(summary.keys()))