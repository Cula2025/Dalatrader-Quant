from __future__ import annotations
from typing import Dict, Any
import pandas as pd

from app.data_providers import get_ohlcv
from app.safebt import run as ENGINE_RUN  # run(df_in, params) -> dict

def _to_series(x) -> pd.Series:
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    try:
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        s = s.sort_index()
    except Exception:
        pass
    return s

def _metrics(ticker: str, eq: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(eq) == 0:
        return out
    total = float(eq.iloc[-1] / eq.iloc[0])
    final = float(eq.iloc[-1])
    rets = eq.pct_change().dropna()
    sharpe = float((rets.mean() / (rets.std() or 1e-12)) * (252 ** 0.5)) if len(rets) else 0.0
    maxdd = float((eq / eq.cummax() - 1.0).min())
    # Buy&Hold på samma fönster
    bhx = 0.0
    try:
        px = get_ohlcv(ticker, start=str(eq.index[0].date()))
        close = pd.to_numeric(pd.Series(px.get("Close")), errors="coerce").dropna()
        close = close[close.index >= eq.index[0]]
        if len(close):
            bhx = float(close.iloc[-1] / close.iloc[0])
    except Exception:
        pass
    out.update(dict(TotalReturn=total, SharpeD=sharpe, MaxDD=maxdd, FinalEquity=final, BuyHold=bhx))
    return out

def run_backtest(p: Dict[str, Any], df=None) -> Dict[str, Any]:
    ticker = (p or {}).get("ticker") or (p or {}).get("symbol") or ""
    params = (p or {}).get("params") or {}
    start  = params.get("start")
    end    = params.get("end")

    # Data om inte given
    if df is None:
        df = get_ohlcv(ticker, start=str(start) if start else None, end=end)

    # *** Viktigt: motorns signatur är run(df_in, params) ***
    out = ENGINE_RUN(df, params)

    # Normalisera equity + fyll metrics
    eq = _to_series(out.get("equity"))
    out["equity"] = eq
    m = dict(out.get("metrics") or {})
    m.setdefault("Trades", len(out.get("trades", [])))
    for k, v in _metrics(ticker, eq).items():
        m.setdefault(k, v)
    out["metrics"] = m
    return out
