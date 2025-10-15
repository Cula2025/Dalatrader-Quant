from __future__ import annotations
from typing import Iterable
import pandas as pd

def buyhold_equity_from_close(
    close: Iterable[float] | pd.Series,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.Series:
    """
    Buy&Hold-equity normaliserad till 1.0 vid start.
    fee_bps + slippage_bps appliceras som engångskostnad vid entry.
    """
    s = pd.Series(close, dtype="float64").dropna()
    if s.empty:
        return pd.Series(name="BH", dtype="float64")
    norm = s / float(s.iloc[0])
    cost = 1.0 - (fee_bps + slippage_bps) / 10000.0
    eq = norm * cost
    eq.name = "BH"
    return eq
