# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict
from app.dfguard import normalize_ohlcv

def _resolve_runner():
    # Försök via btwrap först (om den finns)
    try:
        from app.btwrap import run_backtest as _rb
        return _rb
    except Exception:
        pass
    # Annars plocka modulen och hitta körbar funktion
    from app.portfolio_signals import _import_backtest
    mod = _import_backtest()
    for nm in ("run_backtest","backtest","simulate","run"):
        fn = getattr(mod, nm, None)
        if callable(fn):
            return fn
    if callable(mod):
        return mod
    raise RuntimeError("Hittar ingen backtest-funktion i modulen (run_backtest/backtest/simulate/run).")

def run(df_in, params) -> Dict[str, Any]:
    df = normalize_ohlcv(df_in)
    runner = _resolve_runner()
    return runner(df, params)
