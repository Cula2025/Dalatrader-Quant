from __future__ import annotations
import math, random, json, datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

# ---------- Indikatorer ----------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50.0)

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def macd(close: pd.Series, fast: int=12, slow: int=26, signal:int=9):
    f, s = ema(close, fast), ema(close, slow)
    m = f - s
    sig = ema(m, signal)
    return m, sig, m - sig

def bb_percent_b(close: pd.Series, window=20, nstd=2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper, lower = mid + nstd*std, mid - nstd*std
    return ((close - lower) / (upper - lower)).clip(0, 1)

def atr(df: pd.DataFrame, window=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()

# ---------- Parametrar ----------
@dataclass
class Params:
    use_rsi_filter: bool = True
    rsi_window: int = 14
    rsi_min: float = 25
    rsi_max: float = 60
    use_trend_filter: bool = False
    trend_ma_window: int = 100
    breakout_lookback: int = 0
    exit_lookback: int = 0
    use_macd_filter: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    use_bb_filter: bool = False
    bb_window: int = 20
    bb_nstd: float = 2.0
    bb_min: float = 0.2
    use_stop_loss: bool = False
    stop_mode: str = "pct"   # "pct" | "atr"
    stop_loss_pct: float = 0.08
    atr_window: int = 14
    atr_mult: float = 2.0

# ---------- Logik ----------
def compute_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    x = df.copy()
    x["RSI"] = rsi(x["Close"], p.rsi_window)

    x["TREND"] = ema(x["Close"], p.trend_ma_window) if p.use_trend_filter else np.nan
    if p.use_macd_filter:
        _, _, h = macd(x["Close"], p.macd_fast, p.macd_slow, p.macd_signal)
        x["MACD_H"] = h
    else:
        x["MACD_H"] = 0.0
    x["PCTB"] = bb_percent_b(x["Close"], p.bb_window, p.bb_nstd) if p.use_bb_filter else 0.5

    if p.breakout_lookback > 0:
        x["HH"] = x["High"].rolling(p.breakout_lookback).max()
    else:
        x["HH"] = np.nan
    if p.exit_lookback > 0:
        x["LL"] = x["Low"].rolling(p.exit_lookback).min()
    else:
        x["LL"] = np.nan

    cond_trend = (not p.use_trend_filter) | (x["Close"] > x["TREND"])
    cond_macd = (not p.use_macd_filter) | (x["MACD_H"] > 0)
    cond_bb = (not p.use_bb_filter) | (x["PCTB"] <= p.bb_min)

    rsi_up = (x["RSI"].shift(1) < p.rsi_min) & (x["RSI"] >= p.rsi_min)
    rsi_down = (x["RSI"].shift(1) > p.rsi_max) & (x["RSI"] <= p.rsi_max)

    bo_ok = (p.breakout_lookback == 0) | (x["High"] >= x["HH"])
    ex_bo = (p.exit_lookback == 0) | (x["Low"] <= x["LL"])

    x["BUY"] = rsi_up & cond_trend & cond_macd & cond_bb & bo_ok
    x["SELL"] = rsi_down | ex_bo
    return x

def run_backtest(df: pd.DataFrame, p: Params) -> Dict[str, Any]:
    x = compute_signals(df, p)
    cash, pos, entry_px = 100_000.0, 0.0, 0.0
    eq_curve, trades = [], []
    atr_ser = atr(df, p.atr_window) if p.use_stop_loss and p.stop_mode == "atr" else None

    for i, row in x.iterrows():
        px = float(row["Close"])
        eq_curve.append({"Date": row["Date"], "Equity": cash + pos*px})

        if pos > 0 and p.use_stop_loss:
            if p.stop_mode == "pct":
                sl = entry_px * (1.0 - p.stop_loss_pct)
            else:
                sl = entry_px - p.atr_mult * float(atr_ser.iloc[i])
            if px <= sl:
                cash += pos*px
                trades.append(dict(EntryPrice=entry_px, ExitPrice=px,
                                   EntryTime=entry_time, ExitTime=row["Date"],
                                   PnL=pos*(px-entry_px), reason="StopLoss"))
                pos, entry_px = 0, 0
                continue

        if pos > 0 and row["SELL"]:
            cash += pos*px
            trades.append(dict(EntryPrice=entry_px, ExitPrice=px,
                               EntryTime=entry_time, ExitTime=row["Date"],
                               PnL=pos*(px-entry_px), reason="SignalExit"))
            pos, entry_px = 0, 0
            continue

        if pos == 0 and row["BUY"]:
            pos, entry_px, entry_time = cash/px, px, row["Date"]
            cash = 0.0

    if pos > 0:
        px = float(x.iloc[-1]["Close"])
        cash += pos*px
        trades.append(dict(EntryPrice=entry_px, ExitPrice=px,
                           EntryTime=entry_time, ExitTime=x.iloc[-1]["Date"],
                           PnL=pos*(px-entry_px), reason="EoP"))

    eq_df = pd.DataFrame(eq_curve)
    final_eq = float(eq_df["Equity"].iloc[-1])
    ret = eq_df["Equity"].pct_change().dropna()
    sharpe = float(np.sqrt(252) * ret.mean() / (ret.std() + 1e-9)) if not ret.empty else 0.0
    roll_max = eq_df["Equity"].cummax()
    maxdd = float((eq_df["Equity"]/roll_max - 1).min())
    years = max(1e-9, (df["Date"].iloc[-1] - df["Date"].iloc[0]).days/365)
    cagr = (final_eq/100_000.0)**(1/years) - 1

    return {
        "summary": dict(TotalReturn=final_eq/100000-1, MaxDD=maxdd,
                        SharpeD=sharpe, CAGR=cagr, FinalEquity=final_eq,
                        Trades=len(trades)),
        "equity": eq_df,
        "trades": pd.DataFrame(trades)
    }
