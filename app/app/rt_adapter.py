# app/rt_adapter.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import pandas as pd

# Laddning av OHLCV
def load_ohlcv(source: str, target: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    source: "borsdata" | "csv"
    target: ticker (borsdata) eller s?kv?g (csv)
    """
    if source.lower() == "csv":
        import pandas as pd
        df = pd.read_csv(target, sep=None, engine="python")
        rename = {c: {'date':'Date','open':'Open','high':'High','low':'Low','close':'Close','adj close':'Adj Close','volume':'Volume'}.get(c.lower(),c) for c in df.columns}
        df = df.rename(columns=rename)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        need = [c for c in ["Open","High","Low","Close"] if c in df.columns]
        return df[need]

    elif source.lower() == "borsdata":
        # Anv?nder v?r BD-klient via data_providers
        from app.data_providers import get_ohlcv
        df = get_ohlcv("borsdata", target, start, end)
        need = [c for c in ["Open","High","Low","Close"] if c in df.columns]
        return df[need]

    else:
        raise ValueError(f"Ok?nd k?lla: {source}")


# S?ker k?rning runt backtest (f?ngar fel)
def safe_run_backtest(df: pd.DataFrame, **params) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    from app.backtest_simple import run_backtest
    try:
        return run_backtest(df, **params)
    except Exception as e:
        # returnera tomma f?r UI men med felinfo
        eq = pd.Series(dtype=float)
        trades = pd.DataFrame()
        stats = {"error": str(e)}
        return eq, trades, stats
