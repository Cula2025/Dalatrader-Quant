from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd
from app.data_providers import get_ohlcv

def _resolve_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    return df.sort_index().loc[~df.index.duplicated(keep="last")]

def _pick_close(df: pd.DataFrame) -> pd.Series:
    for c in ("Adj Close","AdjClose","adjusted_close","Close","close"):
        if c in df.columns: 
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 2:
                return s
    # fallback: första numeriska kolumnen
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 2:
            return s
    raise ValueError("Kunde inte hitta pris-kolumn (Close).")

def calc_bh(ticker: str, start: Optional[str], end: Optional[str]) -> Dict[str, Any]:
    df = get_ohlcv(ticker=ticker, start=start, end=end)
    if df is None or getattr(df, "empty", True):
        raise ValueError("Ingen data från Börsdata.")
    df = _resolve_dt_index(df)
    s  = _pick_close(df).dropna()

    ts_start = pd.to_datetime(start) if start else None
    ts_end   = pd.to_datetime(end)   if end   else None

    if ts_start is not None:
        s = s[s.index >= ts_start]
    if ts_end is not None:
        s = s[s.index <= ts_end]
    if s.size < 2:
        raise ValueError("För få datapunkter efter datumklippning.")

    first_dt = s.index[0]; last_dt = s.index[-1]
    first_px = float(s.iloc[0]);   last_px = float(s.iloc[-1])
    bh = (last_px / first_px) - 1.0

    return {
        "bh": bh,
        "first_date": first_dt.date().isoformat(),
        "last_date":  last_dt.date().isoformat(),
        "first_price": first_px,
        "last_price":  last_px,
        "rows_used": int(s.size),
        "ticker": ticker,
        "start_req": start,
        "end_req": end,
    }
