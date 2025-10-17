from __future__ import annotations
import pandas as pd
import numpy as np

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=int(n), adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    r = close.diff()
    up = r.clip(lower=0).rolling(n).mean()
    dn = (-r.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev).abs(),
        (low - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(int(n)).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(high, low, close, 1)
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(n).sum() / tr.rolling(n).sum()
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(n).sum() / tr.rolling(n).sum()
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean().fillna(0)

def choppiness(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = atr(high, low, close, 1)
    sum_tr = tr.rolling(n).sum()
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    denom = (hh - ll).replace(0, np.nan)
    chop = 100 * np.log10(sum_tr / denom) / np.log10(n)
    return chop.replace([np.inf, -np.inf], np.nan).bfill().fillna(100)

def obv_slope(close: pd.Series, volume: pd.Series, n: int = 20) -> pd.Series:
    ret = close.diff().fillna(0)
    obv = (np.sign(ret) * volume.fillna(0)).cumsum()
    return obv.diff(int(n))

def donchian_high(high: pd.Series, n: int) -> pd.Series:
    return high.rolling(int(n)).max()

def donchian_low(low: pd.Series, n: int) -> pd.Series:
    return low.rolling(int(n)).min()

def bb_percent_b(close: pd.Series, n: int = 20, nstd: float = 2.0) -> pd.Series:
    m = close.rolling(int(n)).mean()
    sd = close.rolling(int(n)).std()
    upper = m + nstd * sd
    lower = m - nstd * sd
    return (close - lower) / (upper - lower).replace(0, np.nan)
