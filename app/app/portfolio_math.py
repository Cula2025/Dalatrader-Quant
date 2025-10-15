import pandas as pd
from typing import Any, Iterable, Sequence

def pick_first(*vals):
    """Returnera första 'meningsfulla' värdet (ej None, ej tom pd obj). Undviker pandas truthiness-fällor."""
    for v in vals:
        if v is None:
            continue
        # pandas-objekt: kolla .empty
        if hasattr(v, "empty"):
            try:
                if not v.empty:
                    return v
                else:
                    continue
            except Exception:
                pass
        return v
    return None

def _to_series(x: Any) -> pd.Series:
    """Konvertera olika format (Series/DataFrame/list/dict) till numerisk Series med NaN borttagna."""
    if x is None:
        return pd.Series(dtype="float64")
    if isinstance(x, pd.Series):
        s = pd.to_numeric(x, errors="coerce").dropna()
        return s
    if isinstance(x, pd.DataFrame):
        for col in ("equity","close","Close","price","Price"):
            if col in x.columns:
                return pd.to_numeric(x[col], errors="coerce").dropna()
        if x.shape[1] == 1:
            return pd.to_numeric(x.iloc[:,0], errors="coerce").dropna()
        num = x.select_dtypes(include="number")
        if not num.empty:
            return num.iloc[:,0].dropna()
        return pd.Series(dtype="float64")
    if isinstance(x, (list, tuple)):
        return pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if isinstance(x, dict):
        try:
            return pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        except Exception:
            return pd.Series(dtype="float64")
    return pd.Series(dtype="float64")

def _align_and_normalize(equities: Iterable[pd.Series]) -> pd.DataFrame:
    """Inner-aligna flera serier i tid och normalisera varje kolumn till 1.0 vid start."""
    series = []
    for e in equities:
        s = _to_series(e)
        if s is None or s.empty:
            continue
        if not isinstance(s.index, pd.DatetimeIndex):
            try:
                s.index = pd.to_datetime(s.index)
            except Exception:
                pass
        series.append(s)
    if not series:
        return pd.DataFrame(index=pd.DatetimeIndex([], dtype="datetime64[ns]"))
    df = pd.concat(series, axis=1, join="inner").dropna(how="any")
    if df.empty:
        return df
    first = df.iloc[0]
    df = df.divide(first, axis=1)
    return df

def equal_weight_rebalanced(equities: Sequence[Any]) -> pd.Series:
    """Daglig EW rebalans: medel av normaliserade prisserier."""
    df = _align_and_normalize(equities)
    if df.empty:
        return pd.Series(dtype="float64", name="Rebalanced")
    s = df.mean(axis=1)
    s.name = "Rebalanced"
    return s

def equal_weight_buyhold(equities: Sequence[Any]) -> pd.Series:
    """EW buy&hold (lika insats dag 0): för prisserier blir det medel av P_t/P_0."""
    df = _align_and_normalize(equities)
    if df.empty:
        return pd.Series(dtype="float64", name="Buy&Hold")
    s = df.mean(axis=1)
    s.name = "Buy&Hold"
    return s

def lvl(s):
    try:
        return float(s.iloc[-1])
    except Exception:
        return None
