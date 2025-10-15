from __future__ import annotations
from typing import Any
import pandas as pd

def _to_num_series(x: Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").dropna()

    if isinstance(x, pd.DataFrame):
        for c in ("equity", "Equity", "value", "values"):
            if c in x.columns:
                s = pd.to_numeric(x[c], errors="coerce").dropna()
                if s.size > 1:
                    return s
        for c in x.columns:
            s = pd.to_numeric(x[c], errors="coerce").dropna()
            if s.size > 1:
                return s
        return pd.Series(dtype="float64")

    if isinstance(x, (list, tuple)):
        if x and isinstance(x[0], (int, float)):
            return pd.Series(x, dtype="float64")
        if x and isinstance(x[0], dict):
            for vk in ("y","value","equity","v"):
                s = pd.to_numeric(pd.Series([row.get(vk) for row in x]), errors="coerce").dropna()
                if s.size > 1:
                    return s
        return pd.Series(dtype="float64")

    if isinstance(x, dict) and "values" in x:
        return pd.to_numeric(pd.Series(x["values"]), errors="coerce").dropna()

    try:
        return pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype="float64")

def extract_equity(res: Any) -> pd.Series:
    """
    Extrahera numerisk equity-serie (normaliserad så första värdet = 1.0).
    """
    candidates: list[Any] = []
    if isinstance(res, dict):
        for k in ("equity","equity_curve","equitySeries","Equity","curve","equity_vec"):
            if k in res:
                candidates.append(res[k])
                break
        else:
            candidates.extend(res.values())
    else:
        candidates.append(res)

    for v in candidates:
        s = _to_num_series(v)
        if s.size > 1:
            try:
                s = s.sort_index()
            except Exception:
                pass
            first = float(s.iloc[0]) or 1.0
            if first == 0.0:
                first = 1.0
            return s / first

    raise ValueError("Could not extract numeric equity from backtest result.")
