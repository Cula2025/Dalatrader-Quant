from __future__ import annotations
import pandas as pd

__all__ = ["normalize_ohlcv"]

_COLMAP = {
    "o":"Open", "open":"Open",
    "h":"High", "high":"High",
    "l":"Low",  "low":"Low",
    "c":"Close","close":"Close",
    "v":"Volume","vol":"Volume","volume":"Volume"
}

def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: _COLMAP.get(str(c).strip().lower(), c) for c in df.columns}
    return df.rename(columns=cols)

def _to_date_index(df: pd.DataFrame) -> pd.DataFrame:
    # Använd befintligt index om det är datum, annars 'Date'-kolumn om den finns
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date", drop=True)
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            # sista utväg: försök på en 'date'/'d'-kolumn om den finns
            for k in ("date","d"):
                if k in df.columns:
                    df = df.set_index(k, drop=True)
                    df.index = pd.to_datetime(df.index)
                    break
    df = df.sort_index()
    df.index.name = "Date"
    return df

def normalize_ohlcv(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställ standardiserad OHLCV:
      - Kolumner: Open, High, Low, Close, Volume
      - DatetimeIndex (name='Date'), sorterad
      - Numeriska typer; NaN i core-kolumner droppas
      - Konsistens: Low <= {Open,Close} <= High
    """
    if df_in is None:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    df = pd.DataFrame(df_in).copy()
    df = _rename_cols(df)

    # Behåll endast relevanta kolumner, skapa Volume om saknas
    for k in ("Open","High","Low","Close"):
        if k not in df.columns:
            # försök härleda saknade OHLC från Close (fallback)
            if "Close" in df.columns and k != "Close":
                df[k] = df["Close"]
            else:
                df[k] = pd.NA
    if "Volume" not in df.columns:
        df["Volume"] = 0

    # Datumindex
    df = _to_date_index(df)

    # Numeriskt
    for k in ("Open","High","Low","Close","Volume"):
        df[k] = pd.to_numeric(df[k], errors="coerce")

    # Droppa rader utan Close
    df = df.dropna(subset=["Close"])

    # Konsistens: High = max(O,H,L,C), Low = min(...)
    hi = df[["Open","High","Low","Close"]].max(axis=1)
    lo = df[["Open","High","Low","Close"]].min(axis=1)
    df["High"] = hi
    df["Low"]  = lo

    # Fyll eventuella NaN i Volume
    df["Volume"] = df["Volume"].fillna(0).astype("int64", errors="ignore")

    # Säkerställ ordning
    df = df[["Open","High","Low","Close","Volume"]]
    return df
