from __future__ import annotations
import re, inspect, importlib
from typing import Any, Dict, Optional, Tuple, List
import pandas as pd
import numpy as np

def _get_bors_loader():
    for m,a in [
        ("app.borsdata","load_df"),
        ("app.borsdata","load_ohlc"),
        ("app.dataproviders.borsdata","load_df"),
        ("app.data","borsdata_load_df"),
    ]:
        try:
            mod = importlib.import_module(m)
            f = getattr(mod, a, None)
            if callable(f):
                return lambda s: f(symbol=s)
        except Exception:
            pass
    for m,a in [
        ("app.__init__", "_load_df_any_alias"),
        ("app.btwrap",   "_load_df_any_alias"),
    ]:
        try:
            mod = importlib.import_module(m)
            f = getattr(mod, a, None)
            if callable(f):
                return lambda s: f(symbol=s)
        except Exception:
            pass
    return None

_BD_LOAD = _get_bors_loader()

def _ticker_variants_borsdata(t: str) -> List[str]:
    b=(t or "").strip()
    u=re.sub(r"\s+"," ", b).upper()
    cands=[]
    for base in (b, u):
        for sep in [" ","-",".",""]:
            v = base.replace(" ", sep)
            cands.extend([v, v + ".ST"])
    seen=set(); out=[]
    for c in cands:
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out

def load_price_series(ticker: str) -> Optional[pd.Series]:
    if _BD_LOAD is None:
        return None
    for cand in _ticker_variants_borsdata(ticker):
        try:
            df = _BD_LOAD(cand)
            if isinstance(df, pd.DataFrame) and not df.empty:
                col=None
                for c in ("Adj Close","adj_close","Close","close","c"):
                    if c in df.columns: col=c; break
                if col is None:
                    if all(x in df.columns for x in ("open","high","low","close")):
                        col="close"
                    elif all(x in df.columns for x in ("Open","High","Low","Close")):
                        col="Close"
                if not col: 
                    continue
                s = df[col].astype(float).dropna()
                if s.empty: 
                    continue
                s.index = pd.to_datetime(df.index)
                s = s.sort_index()
                return (s / s.iloc[0]).rename(ticker)
        except Exception:
            pass
    return None

_ENGINE_CANDIDATES = [
    ("app.optimizer",             "run_for_ticker"),
    ("app.optimizer",             "run_profile"),
    ("app.optimizer",             "run_single"),
    ("app.portfolio_engine",      "run_for_ticker"),
    ("app.portfolio_backtest",    "run_for_ticker"),
    ("app.backtest",              "run_backtest"),
    ("app.backtester",            "run_single"),
]

def detect_engine() -> Tuple[Optional[Any], str]:
    for m, a in _ENGINE_CANDIDATES:
        try:
            mod = importlib.import_module(m)
            fn  = getattr(mod, a, None)
            if callable(fn):
                return fn, f"{m}.{a}"
        except Exception:
            pass
    return None, ""

def _series_from_result(res: Any) -> Optional[pd.Series]:
    if res is None:
        return None
    s=None
    if isinstance(res, dict):
        for k in ["equity","curve","balance","cumret","portfolio","portfolio_value","equity_curve"]:
            v=res.get(k)
            if v is None: 
                continue
            if isinstance(v, dict):
                try: s=pd.Series(v); break
                except Exception: pass
            if isinstance(v, list) and v:
                try:
                    if isinstance(v[0], dict):
                        kd = next((x for x in ["date","Date","ts","timestamp"] if x in v[0]), None)
                        kv = next((x for x in ["equity","value","val","y","close","Close"] if x in v[0]), None)
                        if kd and kv:
                            idx=pd.to_datetime([x[kd] for x in v])
                            vals=[x[kv] for x in v]
                            s=pd.Series(vals, index=idx); break
                    else:
                        s=pd.Series(v); break
                except Exception: pass
    if s is None and isinstance(res, pd.Series): s=res
    if s is None and isinstance(res, pd.DataFrame):
        for c in ("equity","value","val","close","Close"):
            if c in res.columns:
                try: s=res[c]; break
                except Exception: pass
    if s is None: 
        return None
    s = s.dropna()
    if len(s)>0 and s.iloc[0]>0:
        s = s / float(s.iloc[0])
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    return s.sort_index()

def get_strategy_curve(ticker: str, profile: Optional[Dict[str,Any]]=None, profile_index: Optional[int]=None) -> Optional[pd.Series]:
    fn, name = detect_engine()
    if fn is None:
        return None
    try:
        sig = inspect.signature(fn)
        kwargs = {}
        if "ticker" in sig.parameters: kwargs["ticker"] = ticker
        elif "symbol" in sig.parameters: kwargs["symbol"] = ticker
        elif "code"   in sig.parameters: kwargs["code"]   = ticker

        for key in ["profile", "params", "config", "settings"]:
            if key in sig.parameters and profile is not None:
                kwargs[key] = profile

        if "profile_index" in sig.parameters and profile_index is not None:
            kwargs["profile_index"] = profile_index

        res = fn(**kwargs) if kwargs else fn(ticker)
        return _series_from_result(res)
    except Exception:
        return None

def build_portfolio(curves: Dict[str,pd.Series], weights: Optional[Dict[str,float]]=None) -> Optional[pd.Series]:
    if not curves:
        return None
    idx = sorted(set().union(*[set(s.index) for s in curves.values()]))
    df = pd.DataFrame({k: v.reindex(idx).ffill() for k,v in curves.items()})
    df = df.dropna(how="all")
    if df.empty:
        return None
    if not weights:
        w = np.ones(df.shape[1]) / df.shape[1]
    else:
        ww = np.array([max(0.0, float(weights.get(k, 0.0))) for k in df.columns], dtype=float)
        if ww.sum() <= 0: ww = np.ones_like(ww)
        w = ww / ww.sum()
    port = (df * w).sum(axis=1)
    if len(port)>0 and port.iloc[0] != 0:
        port = port / port.iloc[0]
    return port.rename("Portfölj")
