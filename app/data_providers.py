from __future__ import annotations
from typing import Optional
import pandas as pd, numpy as np

def get_ohlcv(ticker: str, start: Optional[str]=None, end: Optional[str]=None, freq: str="B") -> pd.DataFrame:
    s = pd.to_datetime(start) if start else pd.Timestamp("2020-01-01")
    e = pd.to_datetime(end) if end else pd.Timestamp.today().normalize()
    if s>=e: return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    idx = pd.bdate_range(s, e, freq=freq)
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    rets = rng.normal(0.0003, 0.01, len(idx))
    close = 100.0 * (1.0 + rets).cumprod()
    df = pd.DataFrame(index=idx)
    df["Open"]=close; df["High"]=close*1.003; df["Low"]=close*0.997; df["Close"]=close; df["Volume"]=1000
    df.index.name="Date"; return df
