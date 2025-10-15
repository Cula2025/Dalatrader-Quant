from __future__ import annotations
from typing import Any, Iterable, List, Dict
import pandas as pd
import numpy as np

# Vi försöker hålla dessa kolumner där de finns
CANON_ORDER = ["side","price","qty","value","ticker","profile","cash","equity","pnl"]

def _to_day(x) -> pd.Timestamp:
    """Till dagsnivå (UTC-naiv)."""
    return pd.to_datetime(x, errors="coerce").normalize()

def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Sortera, indexera på datum och returnera bara kända kolumner (om de finns)."""
    if "date" in df.columns:
        df = df.copy()
        df["date"] = df["date"].map(_to_day)
        df = df.dropna(subset=["date"]).set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Kan inte inferera datum -> returnera tomt hellre än felaktigt (1970-problem)
        return pd.DataFrame(columns=[c for c in CANON_ORDER if c in df.columns])

    # Se till att side finns i rätt ordning om den finns
    if "side" in df.columns:
        cat = pd.CategoricalDtype(categories=["SELL","BUY"], ordered=True)
        try:
            df["side"] = df["side"].astype(cat)
        except Exception:
            pass
        df = df.sort_values(by=["date","side"] if "date" in df.columns else ["side","price"], kind="stable")
    else:
        df = df.sort_index(kind="stable")

    # Sätt tydliga typer där möjligt (utan att kasta bort data)
    for col in ("price","qty","value","cash","equity","pnl"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cols = [c for c in CANON_ORDER if c in df.columns]
    return df[cols] if cols else df

def _from_entry_exit_df(tr: pd.DataFrame,
                        ticker: str | None = None,
                        profile: str | None = None) -> pd.DataFrame:
    """Konvertera DF med Entry/Exit-kolumner till BUY/SELL-rader.
    Vi sätter inte qty/value (lämnas NaN) – portföljlogik sköter sizing."""
    req = {"EntryTime","EntryPrice","ExitTime","ExitPrice"}
    if not req.issubset(set(tr.columns)):
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, r in tr.iterrows():
        et, ep = r["EntryTime"], r["EntryPrice"]
        xt, xp = r["ExitTime"],  r["ExitPrice"]
        if pd.notna(et) and pd.notna(ep):
            rows.append({"date": _to_day(et), "side":"BUY",  "price": float(ep),
                         "ticker": ticker, "profile": profile})
        if pd.notna(xt) and pd.notna(xp):
            rows.append({"date": _to_day(xt), "side":"SELL", "price": float(xp),
                         "ticker": ticker, "profile": profile,
                         "pnl": (float(r["PnL"]) if "PnL" in tr.columns else np.nan)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return _finish(df)

def _normalize_ledger_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ledger-lik DF (side/price, ev. 'date'-kolumn)."""
    if "side" not in df.columns or "price" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    if "date" not in d.columns and isinstance(d.index, pd.DatetimeIndex):
        d = d.rename_axis("date").reset_index()
    elif "date" not in d.columns:
        # ingen datuminformation -> ge upp (hellre tomt)
        return pd.DataFrame()
    return _finish(d)

def to_trades_df(obj: Any,
                 ticker: str | None = None,
                 profile: str | None = None) -> pd.DataFrame:
    """Gör om olika 'trades'-strukturer till en kanoniserad BUY/SELL-DataFrame.

    Accepterar:
      - dict med 'trades'
      - DataFrame med Entry/Exit-kolumner
      - redan ledger (side/price + date)
      - list av dicts (försöksvis till DF)
    """
    # 1) dict med trades
    if isinstance(obj, dict):
        if "trades" in obj:
            return to_trades_df(obj["trades"], ticker=ticker, profile=profile)
        # ibland ligger metadata i dicten – försök DF
        try:
            df = pd.DataFrame(obj)
            # Välj bara relevanta kolumner om möjligt
            if set(("EntryTime","EntryPrice","ExitTime","ExitPrice")).issubset(df.columns):
                return _from_entry_exit_df(df, ticker, profile)
            if set(("side","price")).issubset(df.columns):
                return _normalize_ledger_df(df)
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame()

    # 2) DataFrame
    if isinstance(obj, pd.DataFrame):
        cols = set(obj.columns)
        if {"EntryTime","EntryPrice","ExitTime","ExitPrice"}.issubset(cols):
            return _from_entry_exit_df(obj, ticker, profile)
        if {"side","price"}.issubset(cols) or "date" in cols or isinstance(obj.index, pd.DatetimeIndex):
            return _normalize_ledger_df(obj)
        # okänt format
        return pd.DataFrame()

    # 3) lista av dictar
    if isinstance(obj, list):
        try:
            df = pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame()
        return to_trades_df(df, ticker=ticker, profile=profile)

    # annars tomt
    return pd.DataFrame()
