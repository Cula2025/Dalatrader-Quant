from __future__ import annotations
import math, random, json, datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app.data_providers import get_ohlcv


# ---------- Hjälp: data ----------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(df).copy()
    if "Date" not in x.columns:
        idx = x.index.name or "Date"
        x = x.reset_index().rename(columns={idx: "Date"})
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
    for c in ("Open","High","Low","Close","Volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return x


# ---------- Indikatorer ----------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down.replace(0, np.nan))
    out = 100 - 100/(1+rs)
    return out.fillna(50.0)

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def macd(close: pd.Series, fast: int=12, slow: int=26, signal:int=9) -> Tuple[pd.Series,pd.Series,pd.Series]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    m = ema_fast - ema_slow
    s = ema(m, signal)
    h = m - s
    return m, s, h

def bb_percent_b(close: pd.Series, window:int=20, nstd:float=2.0) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = mid + nstd*std
    lower = mid - nstd*std
    pctb = (close - lower) / (upper - lower)
    return pctb.clip(0,1)

def atr(df: pd.DataFrame, window:int=14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# ---------- Backtest (enkel long, 100% per affär) ----------
@dataclass
class Params:
    # RSI
    use_rsi_filter: bool = True
    rsi_window: int = 14
    rsi_min: float = 25.0
    rsi_max: float = 60.0
    # Trend-gate
    use_trend_filter: bool = False
    trend_ma_type: str = "EMA"
    trend_ma_window: int = 100
    # Breakout/Exit (valfritt, enkel Donchian)
    breakout_lookback: int = 0
    exit_lookback: int = 0
    # MACD
    use_macd_filter: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Bollinger %B
    use_bb_filter: bool = False
    bb_window: int = 20
    bb_nstd: float = 2.0
    bb_min: float = 0.0  # t.ex. mean-reversion: kräv %B < bb_min
    # Stop-loss
    use_stop_loss: bool = False
    stop_mode: str = "pct"  # "pct" | "atr"
    stop_loss_pct: float = 0.08
    atr_window: int = 14
    atr_mult: float = 2.0

def compute_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    x = df.copy()

    # Basindikatorer
    x["RSI"] = rsi(x["Close"], p.rsi_window)
    if p.use_trend_filter and p.trend_ma_window > 0:
        x["TREND"] = ema(x["Close"], p.trend_ma_window)
    else:
        x["TREND"] = np.nan

    if p.use_macd_filter:
        m, s, h = macd(x["Close"], p.macd_fast, p.macd_slow, p.macd_signal)
        x["MACD_H"] = h
    else:
        x["MACD_H"] = 0.0

    if p.use_bb_filter:
        x["PCTB"] = bb_percent_b(x["Close"], p.bb_window, p.bb_nstd)
    else:
        x["PCTB"] = 0.5

    # Donchian
    if p.breakout_lookback > 0:
        x["HH"] = x["High"].rolling(p.breakout_lookback).max()
    else:
        x["HH"] = np.nan
    if p.exit_lookback > 0:
        x["LL"] = x["Low"].rolling(p.exit_lookback).min()
    else:
        x["LL"] = np.nan

    # Entry/Exit villkor
    cond_trend = (~p.use_trend_filter) | (x["Close"] > x["TREND"])
    cond_macd  = (~p.use_macd_filter)  | (x["MACD_H"] > 0)
    cond_bb    = (~p.use_bb_filter)    | (x["PCTB"] <= p.bb_min)

    # RSI korsningar
    rsi_up   = (x["RSI"].shift(1) < p.rsi_min) & (x["RSI"] >= p.rsi_min)
    rsi_down = (x["RSI"].shift(1) > p.rsi_max) & (x["RSI"] <= p.rsi_max)

    # Breakout
    bo_ok   = (p.breakout_lookback == 0) | (x["High"] >= x["HH"])
    ex_bo   = (p.exit_lookback == 0)     | (x["Low"]  <= x["LL"])

    x["BUY"]  = rsi_up   & cond_trend & cond_macd & cond_bb & bo_ok
    x["SELL"] = rsi_down | ex_bo
    return x

def run_backtest(df: pd.DataFrame, p: Params) -> Dict[str, Any]:
    x = compute_signals(df, p)
    cash = 100_000.0
    pos  = 0.0
    entry_px = 0.0
    eq_curve = []
    trades: List[Dict[str, Any]] = []

    # Stop-loss nivåer
    cur_stop = np.nan

    atr_ser = atr(df, p.atr_window) if p.use_stop_loss and p.stop_mode == "atr" else None

    for i, row in x.iterrows():
        px = float(row["Close"])

        # uppdatera equity
        eq = cash + pos*px
        eq_curve.append({"Date": row["Date"], "Equity": eq})

        # Stop-loss check
        if pos > 0 and p.use_stop_loss:
            if p.stop_mode == "pct":
                sl = entry_px * (1.0 - p.stop_loss_pct)
            else:  # ATR
                a = float(atr_ser.iloc[i]) if atr_ser is not None else 0.0
                sl = entry_px - p.atr_mult * a
            if px <= sl and pos > 0:
                # stäng
                cash += pos*px
                trades.append({
                    "EntryTime": entry_time, "EntryPrice": entry_px,
                    "ExitTime": row["Date"], "ExitPrice": px,
                    "PnL": pos*(px-entry_px), "reason": "StopLoss"
                })
                pos = 0.0
                entry_px = 0.0
                continue  # nästa bar

        # Exit
        if pos > 0 and bool(row["SELL"]):
            cash += pos*px
            trades.append({
                "EntryTime": entry_time, "EntryPrice": entry_px,
                "ExitTime": row["Date"], "ExitPrice": px,
                "PnL": pos*(px-entry_px), "reason": "SignalExit"
            })
            pos = 0.0
            entry_px = 0.0
            continue

        # Entry
        if pos == 0 and bool(row["BUY"]):
            pos = cash / px   # 100% all-in
            entry_px = px
            entry_time = row["Date"]
            cash = 0.0

    # stäng vid slutet om öppen
    if pos > 0:
        px = float(x.iloc[-1]["Close"])
        cash += pos*px
        trades.append({
            "EntryTime": entry_time, "EntryPrice": entry_px,
            "ExitTime": x.iloc[-1]["Date"], "ExitPrice": px,
            "PnL": pos*(px-entry_px), "reason": "EoP"
        })
        pos = 0.0

    eq_df = pd.DataFrame(eq_curve).dropna()
    final_eq = float(eq_df["Equity"].iloc[-1]) if not eq_df.empty else 100_000.0
    tr_dec = (final_eq/100_000.0) - 1.0

    # Buy&Hold som kontroll
    c0, c1 = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
    bh = (c1/c0) - 1.0

    # MaxDD (enkel på equity)
    roll_max = eq_df["Equity"].cummax() if not eq_df.empty else pd.Series([100_000.0])
    dd = (eq_df["Equity"] / roll_max) - 1.0 if not eq_df.empty else pd.Series([0.0])
    maxdd = float(dd.min()) if not dd.empty else 0.0

    # Sharpe ungefär (daglig)
    ret = eq_df["Equity"].pct_change().dropna()
    sharpe_d = float(np.sqrt(252) * (ret.mean() / (ret.std() + 1e-9))) if not ret.empty else 0.0

    # CAGR
    years = max(1e-9, (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25)
    cagr = (final_eq/100_000.0) ** (1/years) - 1.0

    return {
        "summary": {
            "Bars": len(df),
            "Trades": len(trades),
            "TotalReturn": tr_dec,
            "MaxDD": maxdd,
            "SharpeD": sharpe_d,
            "BuyHold": bh,
            "FinalEquity": final_eq,
            "CAGR": cagr
        },
        "equity": eq_df,
        "trades": pd.DataFrame(trades)
    }


# ---------- UI ----------
st.set_page_config(page_title="Dalatrader – Optimizer", layout="wide")
st.title("Optimizer (RSI + valbara filter)")

today = dt.date.today()
colA, colB, colC = st.columns(3)
with colA:
    ticker = st.text_input("Ticker", "VOLV B")
with colB:
    from_date = st.text_input("Från (YYYY-MM-DD)", (today - dt.timedelta(days=365*5)).strftime("%Y-%m-%d"))
with colC:
    to_date = st.text_input("Till (YYYY-MM-DD)", today.strftime("%Y-%m-%d"))

st.subheader("Parametrar (baseline för sök)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    rsi_w = st.number_input("RSI-fönster", 5, 50, 14)
    rsi_min = st.number_input("RSI min (köp-kors upp)", 5.0, 45.0, 25.0, step=0.5)
    rsi_max = st.number_input("RSI max (sälj-kors ned)", 55.0, 90.0, 60.0, step=0.5)
with c2:
    use_trend = st.checkbox("Använd EMA trend-gate", value=True)
    trend_win = st.number_input("EMA-fönster", 5, 250, 100)
with c3:
    use_macd = st.checkbox("MACD-filter (histogram > 0)", value=False)
    macd_fast = st.number_input("MACD fast", 3, 24, 12)
    macd_slow = st.number_input("MACD slow", 10, 50, 26)
    macd_sig  = st.number_input("MACD signal", 3, 24, 9)
with c4:
    use_bb = st.checkbox("Bollinger %B-filter", value=False)
    bb_win = st.number_input("BB fönster", 10, 60, 20)
    bb_std = st.number_input("BB std", 1.0, 3.5, 2.0, step=0.1)
    bb_min = st.number_input("%B ≤ (entry-tak)", 0.0, 1.0, 0.2, step=0.05)

st.subheader("Stop-loss")
c5, c6, c7 = st.columns(3)
with c5:
    use_sl = st.checkbox("Aktivera stop-loss", value=False)
with c6:
    sl_mode = st.selectbox("Stop-mod", ["pct", "atr"], index=0)
with c7:
    sl_pct = st.number_input("Stop % (vid pct)", 0.01, 0.5, 0.08, step=0.01)
atr_win = st.number_input("ATR fönster", 5, 50, 14)
atr_mult = st.number_input("ATR multipel", 0.5, 5.0, 2.0, step=0.5)

st.subheader("Optimering")
sims = st.number_input("Antal simuleringar per profil", 100, 50000, 2000, step=100)
seed = st.number_input("Slumpfrö", 0, 999999, 42)
btn_fetch, btn_bt, btn_opt3 = st.columns(3)

state = st.session_state
state.setdefault("df", pd.DataFrame())
state.setdefault("profiles", {})

def fetch_df() -> pd.DataFrame:
    df = get_ohlcv(ticker=ticker, start=from_date, end=to_date, source="borsdata")
    return normalize_df(df)

def make_params() -> Params:
    return Params(
        use_rsi_filter=True, rsi_window=int(rsi_w), rsi_min=float(rsi_min), rsi_max=float(rsi_max),
        use_trend_filter=bool(use_trend), trend_ma_type="EMA", trend_ma_window=int(trend_win),
        breakout_lookback=55, exit_lookback=20,
        use_macd_filter=bool(use_macd), macd_fast=int(macd_fast), macd_slow=int(macd_slow), macd_signal=int(macd_sig),
        use_bb_filter=bool(use_bb), bb_window=int(bb_win), bb_nstd=float(bb_std), bb_min=float(bb_min),
        use_stop_loss=bool(use_sl), stop_mode=sl_mode, stop_loss_pct=float(sl_pct),
        atr_window=int(atr_win), atr_mult=float(atr_mult)
    )

with btn_fetch:
    if st.button("Hämta data", type="primary"):
        try:
            df = fetch_df()
            state["df"] = df
            st.success(f"Läste {len(df)} rader.")
            st.dataframe(df.tail(10), width="stretch")
        except Exception as e:
            st.error(f"Kunde inte ladda/städa data: {e}")

with btn_bt:
    if st.button("Kör backtest (baseline)"):
        df = state.get("df", pd.DataFrame())
        if df.empty:
            st.warning("Hämta data först.")
        else:
            p = make_params()
            res = run_backtest(df, p)
            sm = res["summary"]
            colm = st.columns(5)
            colm[0].metric("TotalReturn", f"{sm['TotalReturn']*100:.2f}%")
            colm[1].metric("Buy&Hold", f"{sm['BuyHold']*100:.2f}%")
            colm[2].metric("MaxDD", f"{sm['MaxDD']*100:.2f}%")
            colm[3].metric("CAGR", f"{sm['CAGR']*100:.2f}%")
            colm[4].metric("Trades", f"{int(sm['Trades'])}")

            st.line_chart(res["equity"].set_index("Date")["Equity"])
            if not res["trades"].empty:
                show_cols = ["EntryTime","EntryPrice","ExitTime","ExitPrice","PnL","reason"]
                st.dataframe(res["trades"][show_cols].sort_values("EntryTime", ascending=False), width="stretch")

# ------ Optimering (3 profiler) ------
def random_params(base: Params, rng: random.Random) -> Params:
    p = Params(**base.__dict__)
    p.rsi_window = rng.randint(7, 30)
    p.rsi_min = rng.uniform(10, 35)
    p.rsi_max = rng.uniform(55, 85)
    if p.rsi_max - p.rsi_min < 10:
        p.rsi_max = p.rsi_min + 10

    # slumpa filters på/av
    p.use_trend_filter = rng.random() < 0.7
    p.trend_ma_window = rng.randint(50, 180)

    p.use_macd_filter = rng.random() < 0.5
    p.macd_fast = rng.randint(8, 14)
    p.macd_slow = rng.randint(20, 34)
    p.macd_signal = rng.randint(6, 12)

    p.use_bb_filter = rng.random() < 0.5
    p.bb_window = rng.randint(14, 28)
    p.bb_nstd = rng.uniform(1.5, 2.5)
    p.bb_min = rng.uniform(0.1, 0.5)

    # stop
    p.use_stop_loss = rng.random() < 0.5
    p.stop_mode = "atr" if rng.random() < 0.5 else "pct"
    p.stop_loss_pct = rng.uniform(0.05, 0.15)
    p.atr_window = rng.randint(10, 20)
    p.atr_mult = rng.uniform(1.5, 3.0)
    return p

def score_conservative(sm: Dict[str, Any]) -> float:
    return float(sm["TotalReturn"]) - 0.8*abs(sm["MaxDD"]) + 0.0005*sm["Trades"]

def score_balanced(sm: Dict[str, Any]) -> float:
    return float(sm["TotalReturn"]) - 0.5*abs(sm["MaxDD"]) + 0.001*sm["Trades"]

def score_aggressive(sm: Dict[str, Any]) -> float:
    return float(sm["TotalReturn"]) - 0.2*abs(sm["MaxDD"]) + 0.0015*sm["Trades"]

def optimize(df: pd.DataFrame, base: Params, sims: int, seed: int, scorer) -> Tuple[Params, Dict[str,Any]]:
    rng = random.Random(seed)
    best = None
    best_res = None
    best_s = -1e18
    prog = st.progress(0.0, text="Optimerar…")
    for i in range(1, sims+1):
        p = random_params(base, rng)
        try:
            res = run_backtest(df, p)
            sm = res["summary"]
            s = scorer(sm)
            if s > best_s:
                best_s, best, best_res = s, p, res
        except Exception:
            pass
        if i % max(1, sims//100) == 0:
            prog.progress(i/sims, text=f"Optimerar… {i}/{sims}")
    prog.empty()
    return best, best_res

with btn_opt3:
    if st.button("Kör 3 profiler och spara"):
        df = state.get("df", pd.DataFrame())
        if df.empty:
            st.warning("Hämta data först.")
        else:
            base = make_params()
            p1, r1 = optimize(df, base, int(sims), int(seed), score_conservative)
            p2, r2 = optimize(df, base, int(sims), int(seed)+1, score_balanced)
            p3, r3 = optimize(df, base, int(sims), int(seed)+2, score_aggressive)

            results = {"conservative": (p1, r1), "balanced": (p2, r2), "aggressive": (p3, r3)}
            state["profiles"] = results

            for label, it in results.items():
                if it[0] is None:
                    st.warning(f"{label}: Ingen giltig träff.")
                    continue
                st.success(f"{label}: OK")
                st.json(it[0].__dict__)
                sm = it[1]["summary"]
                st.write({k: (f"{v*100:.2f}%" if k in ("TotalReturn","MaxDD","BuyHold","CAGR") else v)
                          for k,v in sm.items()})

            # Spara
            out_dir = Path(__file__).resolve().parents[1] / "outputs" / "opt_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{ticker.replace(' ','_')}_best_backtrack.json"
            payload = {"profiles":[]}
            for label, it in results.items():
                if it[0] is None: 
                    continue
                p, res = it
                sm = res["summary"]
                payload["profiles"].append({
                    "name": f"{ticker} – {label}",
                    "ticker": ticker,
                    "params": p.__dict__,
                    "metrics": {
                        "TotalReturn": sm["TotalReturn"],
                        "MaxDD": sm["MaxDD"],
                        "SharpeD": sm["SharpeD"],
                        "BuyHold": sm["BuyHold"],
                        "CAGR": sm["CAGR"]
                    }
                })
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"Sparat → {out_path}")








