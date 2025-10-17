from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np
from app_v3.data_provider_v3 import get_ohlcv
from app_v3.indicators_v3 import (
    ema, sma, rsi, atr, adx, choppiness, obv_slope,
    donchian_high, donchian_low
)

START_EQUITY = 100_000.0

def _buyhold_factor(close: pd.Series) -> float:
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < 2:
        return 1.0
    return float(s.iloc[-1] / s.iloc[0])

def _metrics_from_equity(eq: pd.Series, bhx: float, trades: int, start: str, end: str) -> Dict[str, float]:
    m: Dict[str, float] = {}
    if len(eq) >= 2:
        m["FinalEquity"]  = float(eq.iloc[-1])
        m["TotalReturn"]  = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        rets = eq.pct_change().dropna()
        if len(rets) > 1 and rets.std() > 0:
            m["SharpeD"] = float(rets.mean() / rets.std() * (252 ** 0.5))
        else:
            m["SharpeD"] = 0.0
        rollmax = eq.cummax()
        dd = (eq / rollmax - 1.0).replace([np.inf, -np.inf], np.nan)
        m["MaxDD"] = float(dd.min())
        days = (pd.to_datetime(end or eq.index.max()) - pd.to_datetime(start or eq.index.min())).days
        years = max(days / 365.25, 1/365.25)
        m["CAGR"] = float((eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1.0)
        m["Bars"] = int(len(eq))
        m["Trades"] = int(trades)
        m["BuyHold"] = float(bhx)  # faktor
    return m

def run_backtest(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    df: OHLCV (Open,High,Low,Close,Volume) med DatetimeIndex
    params: se opt_core_v3.make_params()
    """
    if df is None or len(df) == 0:
        t = params.get("ticker","")
        df = get_ohlcv(t, start=params.get("start"), end=params.get("end"))
    df = df.copy().sort_index()
    o,h,l,c,v = (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    start, end = params.get("start"), params.get("end")
    if start: df = df.loc[pd.to_datetime(start):]
    if end:   df = df.loc[:pd.to_datetime(end)]
    o,h,l,c,v = (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])

    # --- indikatorer (beräknas vid behov) ---
    atr_n = int(params.get("atr_window",14))
    atr_s = atr(h,l,c, atr_n)
    rsi_s = rsi(c, int(params.get("rsi_window",14))) if params.get("use_rsi_filter") else None
    adx_s = adx(h,l,c, 14) if params.get("use_adx_filter") else None
    chop_s= choppiness(h,l,c, 14) if params.get("use_chop_filter") else None
    obv_sl= obv_slope(c, v, int(params.get("obv_slope_window",20))) if params.get("use_obv_filter") else None

    # trend/breakout bas
    mode = params.get("entry_mode","donchian")  # 'donchian' | 'ma_trend' | 'ma_cross'
    fast_n = int(params.get("fast",12))
    slow_n = int(params.get("slow",50))
    if mode == "donchian":
        up = donchian_high(h, int(params.get("breakout_lookback",55))).shift(1)
        dn = donchian_low(l, int(params.get("exit_lookback",25))).shift(1)
    elif mode == "ma_trend":
        base = ema(c, slow_n) if params.get("trend_ma_type","EMA") == "EMA" else sma(c, slow_n)
    else: # ma_cross
        f = ema(c, fast_n)
        s = ema(c, slow_n)
        cross_up = (f > s) & (f.shift(1) <= s.shift(1))
        cross_dn = (f < s) & (f.shift(1) >= s.shift(1))

    # --- simulering ---
    pos = 0.0
    entry_px = 0.0
    days_in = 0
    max_bars = int(params.get("max_bars_in_trade", 0) or 0)

    equity = pd.Series(START_EQUITY, index=df.index, dtype=float)
    cash   = START_EQUITY
    shares = 0.0
    trades = 0

    for i, dt in enumerate(df.index):
        px = float(c.loc[dt])
        # exits först (skip om ingen position)
        exit_now = False
        if pos > 0:
            # hard stop
            if params.get("use_stop_loss"):
                if params.get("stop_mode","pct") == "pct":
                    stop_pct = float(params.get("stop_loss_pct",0.1))
                    if px <= entry_px * (1 - stop_pct):
                        exit_now = True
                else:
                    mult = float(params.get("atr_mult",2.0))
                    if atr_s.loc[dt] > 0 and px <= entry_px - mult * atr_s.loc[dt]:
                        exit_now = True
            # trailing
            if (not exit_now) and params.get("use_atr_trailing"):
                mult = float(params.get("atr_trail_mult",2.0))
                trail = entry_px - mult * atr_s.loc[dt]
                if px <= trail:
                    exit_now = True
            # kanal/MA-exit
            if (not exit_now):
                if mode == "donchian":
                    if not np.isnan(dn.loc[dt]) and px < dn.loc[dt]:
                        exit_now = True
                elif mode == "ma_trend":
                    ma_ok = (c.loc[dt] >= base.loc[dt])
                    if not ma_ok:
                        exit_now = True
                else: # cross
                    if cross_dn.loc[dt]:
                        exit_now = True
            # tids-exit
            if (not exit_now) and max_bars > 0 and days_in >= max_bars:
                exit_now = True

        # entry villkor
        enter_now = False
        if pos == 0:
            # bas
            if mode == "donchian":
                if not np.isnan(up.loc[dt]) and px > up.loc[dt]:
                    enter_now = True
            elif mode == "ma_trend":
                ma_ok = (c.loc[dt] >= base.loc[dt]) and (base.loc[dt] >= base.shift(1).loc[dt])
                if ma_ok:
                    enter_now = True
            else:
                if cross_up.loc[dt]:
                    enter_now = True

            # filter
            if enter_now and params.get("use_rsi_filter"):
                rv = rsi_s.loc[dt]
                if not (float(params.get("rsi_min",30)) <= rv <= float(params.get("rsi_max",70))):
                    enter_now = False
            if enter_now and params.get("use_adx_filter"):
                if adx_s.loc[dt] < float(params.get("adx_min",20)):
                    enter_now = False
            if enter_now and params.get("use_chop_filter"):
                if choppiness(h,l,c,14).loc[dt] > float(params.get("chop_max",60)):
                    enter_now = False
            if enter_now and params.get("use_obv_filter"):
                if obv_sl.loc[dt] <= 0:
                    enter_now = False

        # utför (nästa bar antas – men här same bar för enkelhet; V3 kan justeras till next-bar fill)
        if enter_now and pos == 0:
            shares = cash // px
            if shares > 0:
                cash -= shares * px
                pos = 1.0
                entry_px = px
                days_in = 0
                trades += 1
        elif exit_now and pos > 0:
            cash += shares * px
            shares = 0.0
            pos = 0.0
            days_in = 0
        else:
            if pos > 0:
                days_in += 1

        equity.loc[dt] = cash + shares * px

    bhx = _buyhold_factor(c.loc[equity.index[0]:equity.index[-1]])
    metrics = _metrics_from_equity(equity, bhx, trades, start or str(equity.index[0].date()), end or str(equity.index[-1].date()))
    return {"equity": equity, "metrics": metrics}
