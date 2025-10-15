from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .portfolio_settings import get

@dataclass
class PanicResult:
    triggered: bool
    reason: str
    metric: float | None
    date: pd.Timestamp | None

def check_daily_change(index_close: pd.Series) -> PanicResult:
    """
    Enkel första version:
      triggar om (Close[t]/Close[t-1]-1) <= threshold (default -0.05)
    Kräver DatetimeIndex och minst 2 rader.
    """
    thr = float(get("panic_rule.threshold", -0.05))
    if not isinstance(index_close, pd.Series) or len(index_close) < 2:
        return PanicResult(False, "not_enough_data", None, None)

    idx = index_close.dropna()
    if len(idx) < 2:
        return PanicResult(False, "not_enough_data", None, None)

    last, prev = float(idx.iloc[-1]), float(idx.iloc[-2])
    if prev == 0:
        return PanicResult(False, "invalid_prev", None, idx.index[-1])

    change = last / prev - 1.0
    trig = change <= thr
    return PanicResult(trig, "daily_change", change, idx.index[-1])

def is_panic(index_close: pd.Series) -> PanicResult:
    trig_type = get("panic_rule.trigger", "daily_change")
    if trig_type == "daily_change":
        return check_daily_change(index_close)
    # plats för intraday_drawdown / rolling_window i nästa fas
    return PanicResult(False, "unknown_trigger", None, None)
