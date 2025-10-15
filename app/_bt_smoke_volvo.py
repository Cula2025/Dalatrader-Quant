import json, inspect
import pandas as pd

from app.data_providers import get_ohlcv
from app.backtest import run_backtest

# Volvo-profilen som gav bra resultat tidigare (din "gamla bra" baseline)
VOLVO_BASELINE = {
    "strategy": "rsi",
    "trend_ma_type": "EMA",
    "use_trend_filter": False,
    "trend_ma_window": 0,
    "fast": 15,
    "slow": 100,
    "use_rsi_filter": True,
    "rsi_window": 14,
    "rsi_min": 25.0,
    "rsi_max": 60.0,
    "breakout_lookback": 55,
    "exit_lookback": 20,
    "use_macd_filter": False,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "macd_mode": "above_zero",
    "use_bb_filter": False,
    "bb_window": 20,
    "bb_nstd": 2.0,
    "bb_mode": "exit_below_mid",
    "bb_percent_b_min": 0.8,
    "atr_window": 14,
    "atr_stop_mult": 0.0,
    "atr_trail_mult": 0.0,
    "cost_bps": 0.0,
    "cash_rate_apy": 0.0,
    "max_positions": 1,
    "per_trade_pct": 100.0,
    "max_exposure_pct": 100.0,
    "slip_bps": 0,
}

# En enkel RSI-variant som fallback (ifall motorn bara använder RSI-parametrar)
SIMPLE_RSI = {
    "use_rsi_filter": True,
    "rsi_window": 14,
    "rsi_min": 30.0,
    "rsi_max": 70.0,
    "breakout_lookback": 55,
    "exit_lookback": 20,
    "cost_bps": 0.0,
    "slip_bps": 0,
}

def filter_kwargs(params: dict) -> dict:
    """Filtrerar till endast de nycklar som run_backtest faktiskt accepterar."""
    sig = inspect.signature(run_backtest)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in params.items() if k in allowed}

def run_case(name: str, params: dict):
    print(f"\n=== Case: {name} ===")
    try:
        df = get_ohlcv("VOLV B", start="2015-01-01", end="2025-09-16", source="borsdata", interval="1d")
    except Exception as e:
        print("❌ get_ohlcv failed:", e)
        return

    print("Rows in DF:", len(df))

    kwargs = filter_kwargs(params)
    print("Accepted kwargs by run_backtest():", sorted(kwargs.keys()))

    try:
        res = run_backtest(df, **kwargs)
    except Exception as e:
        print("❌ run_backtest failed:", e)
        return

    summary = res.get("summary", {})
    trades = res.get("trades", None)

    print("Summary:", json.dumps(summary, indent=2, default=str))
    if isinstance(trades, pd.DataFrame):
        print("Trades rows:", len(trades))
        print(trades.head(5).to_string(index=False))
    else:
        print("Trades dataframe missing or empty:", type(trades))

if __name__ == "__main__":
    print("run_backtest signature ->", inspect.signature(run_backtest))
    run_case("VOLVO_BASELINE (full)", VOLVO_BASELINE)
    run_case("SIMPLE_RSI (fallback)", SIMPLE_RSI)
