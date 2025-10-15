import pandas as pd
import numpy as np
from typing import Any, List

_EQ_KEYS: List[str] = ["equity","Equity","summary","equity_curve","curve"]

def _empty(name: str = "equity") -> pd.Series:
    return pd.Series(dtype="float64", name=name)

def extract_equity(obj: Any) -> pd.Series:
    """
    Robust utvinning av en equity-kurva som pd.Series.
    - Dict: plockar nycklar som 'equity', 'summary', m.fl.
    - DataFrame: väljer 'equity-lik' kolumn (equity/value/nav/cumret/total...),
                 annars sista numeriska kolumn. Sätter datumindex om möjligt.
    - Series: numerisk, städad, försöker datetime-index.
    - List-like: konverteras till Series.
    """
    # 0) Tomt?
    if obj is None:
        return _empty()

    # 1) Dict → plocka payload
    if isinstance(obj, dict):
        for k in _EQ_KEYS:
            v = obj.get(k, None)
            if v is not None:
                obj = v
                break

    # 2) Redan Series?
    if isinstance(obj, pd.Series):
        s = pd.to_numeric(obj, errors="coerce").dropna()
        try:
            idx = pd.to_datetime(s.index, errors="coerce")
            mask = ~idx.isna()
            if mask.any():
                s = s[mask]
                s.index = idx[mask]
        except Exception:
            pass
        if s.name is None:
            s.name = "equity"
        return s

    # 3) DataFrame?
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

        # välj datumkolumn om den finns
        date_col = next((c for c in ["date","Date","time","timestamp"] if c in df.columns), None)

        # numeriska kolumner
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        # equity-lika kolumner
        def _is_equityish(c: Any) -> bool:
            lc = str(c).lower()
            return any(k in lc for k in ["equity","value","nav","cumret","cum_return","pnl_cum","total"])

        pref = [c for c in num_cols if _is_equityish(c)]
        col = pref[0] if pref else (num_cols[-1] if num_cols else None)
        if col is None:
            return _empty()

        s = pd.to_numeric(df[col], errors="coerce").dropna()

        # bygg index
        try:
            if date_col is not None:
                idx = pd.to_datetime(df[date_col], errors="coerce")
            else:
                idx = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            idx = None

        if idx is not None:
            # alignera längd & NaT
            try:
                if len(idx) == len(df[col]):
                    mask = ~idx.isna()
                    if len(mask) == len(s):
                        # s kan vara kortare efter dropna; justera försiktigt
                        # reindexera mot df[col]s “validerade” mask om möjligt
                        s_full = pd.to_numeric(df[col], errors="coerce")
                        s_full.index = idx
                        s = s_full.dropna()
                    else:
                        # fallback: sätt index där möjligt
                        s.index = pd.RangeIndex(len(s))
                    # efter reindexering: filtrera bort NaT
                    if isinstance(s.index, pd.DatetimeIndex):
                        s = s[~s.index.isna()]
                else:
                    # längd matchar inte → försök direkt
                    s.index = idx[:len(s)]
                    if isinstance(s.index, pd.DatetimeIndex):
                        s = s[~s.index.isna()]
            except Exception:
                s.index = pd.RangeIndex(len(s))
        else:
            s.index = pd.RangeIndex(len(s))

        s.name = str(col)
        return s

    # 4) List-like fallback
    try:
        s = pd.Series(obj)
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.RangeIndex(len(s))
        s.name = "equity"
        return s
    except Exception:
        return _empty()
