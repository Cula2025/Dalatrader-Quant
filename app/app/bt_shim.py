from __future__ import annotations
from typing import Any, Dict, Tuple
import inspect
import pandas as pd

# Importera din riktiga motor
from app import backtest as engine

def _call_engine(df: pd.DataFrame, params: Dict[str, Any]):
    """
    Prova flera anropsstilar – olika versioner av motorn kan kräva olika signaturer.
    Returnera (summary, equity, trades) eller {}/None där det saknas.
    """
    tried = []

    # Fall A: kwargs direkt
    try:
        res = engine.run_backtest(df, **params)
        return _normalize_result(res)
    except TypeError as e:
        tried.append(("kwargs", str(e)))

    # Fall B: positional params-dict
    try:
        res = engine.run_backtest(df, params)
        return _normalize_result(res)
    except TypeError as e:
        tried.append(("positional_dict", str(e)))

    # Fall C: namnad parameter 'params'
    try:
        res = engine.run_backtest(df, params=params)  # vissa implementationer
        return _normalize_result(res)
    except TypeError as e:
        tried.append(("named_params", str(e)))

    # Fall D: namnad parameter 'config'
    try:
        res = engine.run_backtest(df, config=params)
        return _normalize_result(res)
    except TypeError as e:
        tried.append(("named_config", str(e)))

    # Sista utväg – exponera felspåren
    raise TypeError(f"run_backtest() accepterade inte våra parametrar. Tried: {tried}")

def _normalize_result(res: Any) -> Tuple[Dict[str, Any], Any, pd.DataFrame | None]:
    """
    Normalisera motorns svar till (summary: dict, equity_any: Any, trades_df: pd.DataFrame|None).
    Täcker både tuple- och dict-varianter.
    """
    summary: Dict[str, Any] = {}
    equity_any: Any = None
    trades_df: pd.DataFrame | None = None

    if isinstance(res, tuple):
        # vanligast: (summary, equity, trades)
        if len(res) >= 1 and isinstance(res[0], dict):
            summary = res[0]
        if len(res) >= 2:
            equity_any = res[1]
        if len(res) >= 3:
            trades = res[2]
            try:
                trades_df = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades)
            except Exception:
                trades_df = None
        return summary, equity_any, trades_df

    if isinstance(res, dict):
        summary = res
        tr = res.get("trades") if res.get("trades") is not None else res.get("Trades")
        if tr is not None:
            try:
                trades_df = tr if isinstance(tr, pd.DataFrame) else pd.DataFrame(tr)
            except Exception:
                trades_df = None
        return summary, None, trades_df

    # okänt format – försök bästa gissning
    try:
        for x in res:
            if isinstance(x, dict) and not summary:
                summary = x
            elif trades_df is None:
                trades_df = pd.DataFrame(x)
    except Exception:
        pass

    return summary, equity_any, trades_df

# --------- Publikt API (som dina sidor kallar) ---------

ACCEPT = {
    # RSI
    "strategy", "use_rsi_filter", "rsi_window", "rsi_min", "rsi_max",
    # MA/exit
    "fast", "slow", "breakout_lookback", "exit_lookback",
    # friktion
    "cost_bps", "slip_bps",
    # ev. övrigt din motor tolererar
    "trend_ma_type", "use_trend_filter", "trend_ma_window",
    "use_macd_filter","macd_fast","macd_slow","macd_signal","macd_mode",
    "use_bb_filter","bb_window","bb_nstd","bb_mode","bb_percent_b_min",
    "atr_window","atr_stop_mult","atr_trail_mult",
    "cash_rate_apy","max_positions","per_trade_pct","max_exposure_pct",
}

DEFAULTS = {
    "strategy": "rsi",
    "use_rsi_filter": True,
    "rsi_window": 14,
    "rsi_min": 25.0,
    "rsi_max": 60.0,
    "fast": 15,
    "slow": 100,
    "breakout_lookback": 55,
    "exit_lookback": 20,
    "cost_bps": 0.0,
    "slip_bps": 0,
}

def run_backtest(df: pd.DataFrame, **kwargs) -> Tuple[Dict[str, Any], Any, pd.DataFrame | None]:
    """
    Din sida kallar bt_shim.run_backtest(df, **kwargs).
    Vi filtrerar/mixar defaults och skickar vidare till motorn i den form den accepterar.
    """
    params = {**DEFAULTS}
    # Ta endast nycklar vi tillåter
    for k, v in kwargs.items():
        if k in ACCEPT:
            params[k] = v

    # Försök anropa motorn robust
    summary, equity_any, trades_df = _call_engine(df, params)

    # Sätt vettiga nollor om motorn saknar nycklar i svaret
    summary.setdefault("Bars", len(df))
    summary.setdefault("Trades", 0 if trades_df is None else int(len(trades_df)))
    summary.setdefault("TotalReturn", summary.get("TotalReturn", 0.0))
    summary.setdefault("MaxDD", summary.get("MaxDD", 0.0))
    summary.setdefault("SharpeD", summary.get("SharpeD", 0.0))
    # Buy&Hold fallback
    if "BuyHold" not in summary and len(df) >= 2:
        try:
            bh = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1.0)
            summary["BuyHold"] = bh
        except Exception:
            pass
    if "FinalEquity" not in summary:
        try:
            tr = float(summary.get("TotalReturn", 0.0))
            summary["FinalEquity"] = 100000 * (1.0 + tr)
        except Exception:
            summary["FinalEquity"] = 100000.0

    return summary, equity_any, trades_df
