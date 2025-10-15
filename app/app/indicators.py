from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["rsi", "ema", "atr"]

def ema(s: pd.Series, window: int) -> pd.Series:
    window = max(1, int(window))
    return s.ewm(span=window, adjust=False, min_periods=window).mean()

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    window = max(1, int(window))
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50.0)

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    window = max(1, int(window))
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean().fillna(method="bfill")

