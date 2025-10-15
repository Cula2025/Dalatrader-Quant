from __future__ import annotations
import math
from typing import Any, Iterable, Optional, Tuple
import pandas as pd

# ---- Hjälpare ---------------------------------------------------------------

def _as_df(obj: Any) -> Optional[pd.DataFrame]:
    """Försök göra om obj till DataFrame utan att kasta."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, (list, tuple)):
        try:
            return pd.DataFrame(list(obj))
        except Exception:
            return None
    if isinstance(obj, dict):
        # Vanliga nycklar där trades kan ligga
        for k in ("trades","Trades","trade_log","TradeLog","orders","fills"):
            if k in obj:
                return _as_df(obj[k])
        # Om dict:en i sig ser ut som en radlista
        try:
            return pd.DataFrame([obj])
        except Exception:
            return None
    return None

def _pick_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sätt ett DatetimeIndex av bästa möjliga kandidat."""
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
    else:
        out = df.copy()
        # Finn en datumkolumn
        dcol = _pick_col(out, "date","datetime","time","timestamp","dt")
        if dcol is not None:
            out[dcol] = pd.to_datetime(out[dcol], errors="coerce", utc=True).dt.tz_localize(None)
            out = out.set_index(dcol)
        else:
            # Försök kasta indexet direkt
            idx = pd.to_datetime(out.index, errors="coerce", utc=True).tz_convert(None)
            out.index = idx
    # Rensa bort NaT och sortera
    if not isinstance(out.index, pd.DatetimeIndex):
        # sista fallback: tomt
        return pd.DataFrame(columns=["Ticker","Side","Qty","Price","CashFlow","Amount"])
    out = out[~out.index.isna()].sort_index()
    out.index.name = "Date"
    return out

# ---- Publik API -------------------------------------------------------------

def extract_trades(x: Any, default_ticker: Optional[str]=None, default_profile: Optional[str]=None) -> pd.DataFrame:
    """
    Normaliserar trades till kolumner: Ticker, Side (BUY/SELL),
    Qty, Price, CashFlow (- köp, + sälj), Amount (abs).
    Index: Date (DatetimeIndex, tz-naiv).
    """
    df = _as_df(x)
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker","Side","Qty","Price","CashFlow","Amount"])  # tom

    # Kandidater för centrala fält
    c_side  = _pick_col(df, "side","action","buy_sell","bs","direction")
    c_qty   = _pick_col(df, "qty","quantity","shares","size","amount")
    c_price = _pick_col(df, "price","fill_price","px","avg_price","rate")
    c_cash  = _pick_col(df, "cash","cash_flow","cashflow","value","notional")
    c_tick  = _pick_col(df, "ticker","symbol","asset","instr","name","code")

    out = df.copy()

    # Sätt ticker
    if c_tick is None:
        out["Ticker"] = default_ticker
    else:
        out["Ticker"] = out[c_tick].astype(str)

    # Side
    side = None
    if c_side is not None:
        side = out[c_side]
    elif c_qty is not None:
        # tecknet på qty avgör
        side = out[c_qty].apply(lambda q: "BUY" if float(q) > 0 else "SELL")
    else:
        side = "BUY"
    out["Side"] = side.where(pd.notna(side), "BUY").astype(str).str.upper().replace({"B":"BUY","S":"SELL","1":"BUY","-1":"SELL"})

    # Qty
    if c_qty is None:
        out["Qty"] = 0.0
    else:
        out["Qty"] = pd.to_numeric(out[c_qty], errors="coerce").fillna(0.0)

    # Price
    if c_price is None:
        # Försök härleda från cash/qty
        if c_cash is not None and c_qty is not None:
            with pd.option_context("mode.use_inf_as_na", True):
                out["Price"] = (pd.to_numeric(out[c_cash], errors="coerce")/out["Qty"]).replace([pd.NA, pd.NaT], 0.0).fillna(0.0).abs()
        else:
            out["Price"] = 0.0
    else:
        out["Price"] = pd.to_numeric(out[c_price], errors="coerce").fillna(0.0).abs()

    # CashFlow (− för köp, + för sälj). Prioritera befintlig kolumn.
    if c_cash is not None:
        cf = pd.to_numeric(out[c_cash], errors="coerce").fillna(0.0)
    else:
        sign = out["Side"].map({"BUY":-1.0,"SELL":+1.0}).fillna(-1.0)
        cf = sign * (out["Qty"] * out["Price"])
    out["CashFlow"] = cf.astype(float)

    out["Amount"] = out["CashFlow"].abs()

    # Gör tidsindex
    out = ensure_datetime_index(out)

    # Lägg profil om den finns
    if default_profile is not None and "Profile" not in out.columns:
        out["Profile"] = default_profile

    # Kolumnordning
    cols = ["Ticker","Side","Qty","Price","CashFlow","Amount"]
    extra = [c for c in out.columns if c not in cols]
    out = out[cols + extra]
    return out
