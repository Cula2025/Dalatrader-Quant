from __future__ import annotations
from typing import Dict, Any
import pandas as pd

def sma(series: pd.Series, n: int) -> pd.Series:
    return pd.Series(series, dtype="float64").rolling(n, min_periods=n).mean()

def run_sma_crossover(close: pd.Series, fast: int, slow: int, fee_bps: float=0.0, slippage_bps: float=0.0) -> Dict[str, Any]:
    close = pd.to_numeric(pd.Series(close), errors="coerce").dropna()
    if close.empty or fast<=0 or slow<=0 or fast>=slow:
        return {"equity": pd.Series(dtype="float64"), "trades": []}
    f,s = sma(close, fast), sma(close, slow)
    pos = (f > s).astype(int).shift(1).fillna(0)  # 1=long, 0=flat
    # Enkel equity: normaliserad till 1.0, dagliga avkastningar med pos
    ret = close.pct_change().fillna(0.0)
    gross = (1.0 + ret*pos).cumprod()
    # Kostnad: antag kostnad vid varje positionväxling
    turns = pos.diff().abs().fillna(0.0)
    cost = (1.0 - (fee_bps+slippage_bps)/10000.0) ** turns.cumsum()
    eq = (gross * cost)
    eq.iloc[0] = 1.0
    eq.name = "Strategy"
    # Trades-tabell (enkel)
    trades = []
    in_pos = False; entry_date=None; entry_px=None
    for i in range(1, len(pos)):
        if not in_pos and pos.iloc[i]==1 and pos.iloc[i-1]==0:
            in_pos=True; entry_date=pos.index[i]; entry_px=close.iloc[i]
        elif in_pos and pos.iloc[i]==0 and pos.iloc[i-1]==1:
            exit_date=pos.index[i]; exit_px=close.iloc[i]
            trades.append({"entry": str(entry_date.date()), "exit": str(exit_date.date()),
                           "entry_px": float(entry_px), "exit_px": float(exit_px),
                           "ret": float((exit_px/entry_px)-1.0)})
            in_pos=False
    return {"equity": eq, "trades": trades}
