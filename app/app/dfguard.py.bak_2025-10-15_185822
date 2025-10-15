# -*- coding: utf-8 -*-
from typing import Any
import pandas as pd

_PAIRS = [
    ("open","Open"), ("high","High"), ("low","Low"),
    ("close","Close"), ("volume","Volume")
]

def _ensure_ohlcv_variants(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    # lowercase -> TitleCase
    for low, cap in _PAIRS:
        if low in cols and cap not in cols:
            df[cap] = df[low]
    cols = set(df.columns)
    # TitleCase -> lowercase
    for low, cap in _PAIRS:
        if cap in cols and low not in cols:
            df[low] = df[cap]
    # gör numeriska där det går
    for low, cap in _PAIRS:
        for c in (low, cap):
            if c in df.columns:
                try:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                except Exception:
                    pass
    return df

def _maybe_extract_df_from_tuple(obj):
    """Om obj är en tuple/list: plocka ut första DataFrame eller liknande."""
    seq = list(obj)
    # 1) direkt DataFrame inuti?
    for it in seq:
        if isinstance(it, pd.DataFrame):
            return it
    # 2) vanliga två-elementscases: (symbol, df) eller (meta, df)
    if len(seq) == 2:
        a, b = seq
        if isinstance(b, pd.DataFrame):
            return b
        # b kan vara ett objekt med to_pandas()
        if hasattr(b, "to_pandas"):
            try:
                return b.to_pandas()
            except Exception:
                pass
        # b kan vara {"data":[...]} etc
        if isinstance(b, dict):
            data = b.get("data")
            if isinstance(data, (list, tuple)):
                try:
                    return pd.DataFrame(list(data))
                except Exception:
                    pass
    # 3) lista av dicts
    if seq and all(isinstance(it, dict) for it in seq):
        return pd.DataFrame(seq)
    return None  # hittade ingen tydlig DF

def to_dataframe(obj: Any) -> pd.DataFrame:
    if obj is None:
        raise TypeError("df is None")

    # redan DataFrame?
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, (list, tuple)):
        extracted = _maybe_extract_df_from_tuple(obj)
        if extracted is not None:
            df = extracted
        else:
            # fallback: låt Pandas försöka
            df = pd.DataFrame(obj)
    elif isinstance(obj, dict):
        # vanligast: antingen en tabell-liknande dict, eller {"data":[...]}
        data = obj.get("data") if hasattr(obj, "get") else None
        if isinstance(data, (list, tuple)):
            df = pd.DataFrame(list(data))
        else:
            df = pd.DataFrame(obj)
    else:
        if hasattr(obj, "to_pandas"):
            df = obj.to_pandas()
        else:
            df = pd.DataFrame(obj)

    # försök datumindex
    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date", drop=True)
        except Exception:
            pass
    elif "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date", drop=True)
        except Exception:
            pass

    df = _ensure_ohlcv_variants(df)

    try:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
    except Exception:
        pass
    return df
