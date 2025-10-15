# -*- coding: utf-8 -*-
import pandas as pd

# Vanliga alias -> önskade rubriker
_ALIASES = {
    "Open":   ["Open","open","o","open_price"],
    "High":   ["High","high","h","high_price"],
    "Low":    ["Low","low","l","low_price"],
    "Close":  ["Close","close","c","adjclose","adj_close","price","last"],
    "Volume": ["Volume","volume","vol","turnover","v"]
}

def ensure_ohlcv(obj):
    """Returnera DataFrame med index som DatetimeIndex och kolumner: Open High Low Close Volume."""
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        df = pd.DataFrame(obj)

    # Försök få tidsindex
    if not isinstance(df.index, pd.DatetimeIndex):
        for dc in ("Date","date","timestamp","time"):
            if dc in df.columns:
                df[dc] = pd.to_datetime(df[dc], errors="coerce", utc=True)
                df = df.set_index(dc)
                break
        else:
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)

    # Sätt rubriker enligt alias
    cols = {c: c for c in df.columns}
    present = set(df.columns)
    for want, cands in _ALIASES.items():
        if want in present:
            continue
        for c in cands:
            if c in present:
                cols[c] = want
                break
    if cols:
        df = df.rename(columns=cols)

    need = ["Open","High","Low","Close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Saknar kolumner: {missing}")
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.sort_index()
    return df[["Open","High","Low","Close","Volume"]]
