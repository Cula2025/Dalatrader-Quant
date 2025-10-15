from __future__ import annotations
import pandas as pd
from typing import Any, Dict, Optional

_DATE_CANDS = ("Date","date","timestamp","time","dt")

def _to_dt(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if s.notna().sum()==0: s = pd.to_datetime(series, errors="coerce", unit="s",  utc=True)
    if s.notna().sum()==0: s = pd.to_datetime(series, errors="coerce", unit="ms", utc=True)
    if s.notna().sum()==0: s = pd.to_datetime(series, errors="coerce", unit="ns", utc=True)
    try:    s = s.dt.tz_convert(None)
    except: s = s.dt.tz_localize(None)
    return s

def trades_df_from_result(res: Dict[str, Any], ticker: Optional[str]=None, profile_name: Optional[str]=None) -> pd.DataFrame:
    trades = res.get("trades") or []
    if isinstance(trades, dict): trades = list(trades.values())
    df = pd.DataFrame(trades)
    if df.empty: return df

    dcol = next((c for c in _DATE_CANDS if c in df.columns), None)
    if dcol is None and isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
    elif dcol is not None:
        s = _to_dt(df[dcol])
        out = df.assign(_dt=s).dropna(subset=["_dt"]).set_index("_dt").sort_index()
    else:
        return pd.DataFrame()  # kan inte tolka datum

    # normalisera några nyckelkolumner
    side  = next((c for c in ("side","Side","action","Action","signal","Signal","type","Type") if c in out.columns), None)
    qty   = next((c for c in ("qty","quantity","Quantity","size","Size","amount","Amount","Shares") if c in out.columns), None)
    price = next((c for c in ("price","Price","fill_price","FillPrice","Close","close","px") if c in out.columns), None)

    if qty   is not None: out["_qty"]   = pd.to_numeric(out[qty],   errors="coerce")
    else:                 out["_qty"]   = 0.0
    if price is not None: out["_price"] = pd.to_numeric(out[price], errors="coerce")
    else:                 out["_price"] = 0.0

    def _dir(x: Any) -> int:
        if isinstance(x, str):
            xs = x.strip().lower()
            if "sell" in xs or xs in ("s","exit","close","reduce"): return 1
            if "buy"  in xs or xs in ("b","entry","open","long","short"): return -1
        return 0
    if side: out["_dir"] = out[side].map(_dir).fillna(0).astype(int)
    else:    out["_dir"] = out["_qty"].apply(lambda q: 1 if q<0 else (-1 if q>0 else 0))

    out["cash_flow"] = out["_dir"] * (out["_qty"].abs() * out["_price"])
    out["value_abs"] = (out["_qty"].abs() * out["_price"])
    if ticker:       out["Ticker"]  = ticker
    if profile_name: out["Profile"] = profile_name
    return out
