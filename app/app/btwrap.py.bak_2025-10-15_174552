from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from app.data_providers import get_ohlcv
from app.btcore import run_sma_crossover

def run_backtest(p: Dict[str, Any], df=None) -> Dict[str, Any]:
    """
    p: {"ticker": str, "params": { "fast": int, "slow": int, "fee_bps": float, "slippage_bps": float }}
    df: valfri OHLCV DataFrame (om du vill köra med färdigdata)
    """
    ticker = p.get("ticker","AAK")
    params = p.get("params",{}) or {}
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 50))
    fee_bps = float(params.get("fee_bps", 5.0))
    slippage_bps = float(params.get("slippage_bps", 5.0))

    if df is None:
        df = get_ohlcv(ticker, start="2020-10-05")
    if not isinstance(df, pd.DataFrame) or "Close" not in df.columns or df.empty:
        return {"equity": pd.Series(dtype="float64"), "trades": []}
    res = run_sma_crossover(df["Close"], fast, slow, fee_bps, slippage_bps)
    # säkerställ tidsindex sorterat
    eq = res["equity"].copy().sort_index()
    return {"equity": eq, "trades": res.get("trades", [])}
