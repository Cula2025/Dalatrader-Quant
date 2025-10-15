# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict
from app.dfguard import normalize_ohlcv

def _resolve_runner():
    from app.portfolio_signals import _import_backtest, _maybe_build_params
    try:
        rb, Params = _import_backtest()  # tuple: (funktion, Params-klass-eller-None)
        if callable(rb):
            def _call(df, params):
                built = _maybe_build_params(Params, params or {})
                return rb(df, built)
            return _call
    except Exception:
        pass

    # Fallback (om tuplevägen inte funkade): försök med modul/standardnamn eller backtest.py
    import importlib.util, os
    # Prova core/backtest via välkända platser
    for path in ('/srv/trader/backtest.py',):
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location('core_backtest', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            for nm in ('run_backtest','backtest','simulate','run'):
                fn = getattr(mod, nm, None)
                if callable(fn):
                    return fn
    raise RuntimeError('Hittar ingen backtest-funktion (tuple/modul/fallback misslyckades).')
def run(df_in, params) -> Dict[str, Any]:
    df = normalize_ohlcv(df_in)
    runner = _resolve_runner()
    return runner(df, params)
