from __future__ import annotations
from typing import Dict, Any, Optional
import math
import pandas as pd
from app.equity_extract import extract_equity

def _bh_col(df: pd.DataFrame) -> str:
    for c in ("Adj Close", "Close", "close", "c"):
        if c in df.columns:
            return c
    # fallback: första numeriska kolumn
    for c in df.columns:
        try:
            float(pd.to_numeric(df[c], errors="coerce").dropna().iloc[0])
            return c
        except Exception:
            pass
    raise ValueError("Ingen pris-kolumn funnen (t.ex. Close/Adj Close).")

def _max_dd(series: pd.Series) -> float:
    """Max drawdown som negativ andel, t.ex. -0.32."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    roll_max = x.cummax()
    dd = x / roll_max - 1.0
    return float(dd.min())

def _sharpe_daily(equity: pd.Series, periods_per_year: int = 252) -> float:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 3:
        return float("nan")
    rets = eq.pct_change().dropna()
    if rets.std(ddof=1) == 0 or math.isnan(rets.std(ddof=1)):
        return float("nan")
    return float((rets.mean() / rets.std(ddof=1)) * math.sqrt(periods_per_year))

def evaluate_candidate(df: pd.DataFrame, params: Dict[str, Any], return_series: bool = False) -> Dict[str, Any]:
    """
    Kör samma backtestmotor som UI:t (fallback-runbacktest med df, params),
    extraherar equity, och beräknar mått.
    """
    if df is None or len(df) < 3:
        raise ValueError("DF saknas eller är för kort för backtest.")

    try:
        from backtest import run_backtest as RUN
    except Exception as e:
        raise RuntimeError(f"Kunde inte importera backtest.run_backtest: {e}")

    res = RUN(df, dict(params or {}))
    eq  = extract_equity(res)  # start=1.0

    col = _bh_col(df)
    P = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(P) < 2:
        raise ValueError("För få datapunkter i prisserien.")

    bh = float(P.iloc[-1]) / float(P.iloc[0]) - 1.0
    tr = float(eq.iloc[-1]) - 1.0

    init = float(params.get("initial_equity", 100_000))
    final_equity = float(eq.iloc[-1]) * init

    out = {
        "TotalReturn": tr,
        "BuyHold": bh,
        "FinalEquity": final_equity,
        "MaxDD": _max_dd(eq),
        "SharpeD": _sharpe_daily(eq),
    }
    if return_series:
        out["equity_series"] = eq
    return out

def pack_profile(ticker: str, flavor_name: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    m = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}
    return {
        "name": f"{ticker} – {flavor_name}",
        "ticker": ticker,
        "params": dict(params or {}),
        "metrics": m,
    }
