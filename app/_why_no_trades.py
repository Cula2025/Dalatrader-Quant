from __future__ import annotations
import inspect, json
from pathlib import Path
import pandas as pd

from app.data_providers import get_ohlcv
from app.backtest import run_backtest

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index":"Date"})
        else:
            # try common date-like headers
            for a in ("date","time","timestamp","DateTime","datetime"):
                if a in df.columns:
                    df = df.rename(columns={a:"Date"})
                    break
    needed = ["Date","Open","High","Low","Close","Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV/DF saknar kolumner. Måste ha: {needed}. Har: {list(df.columns)}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    for c in ("Open","High","Low","Close","Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

def accepted_kwargs():
    sig = inspect.signature(run_backtest)
    return [p.name for p in sig.parameters.values() if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)]

def try_case(df: pd.DataFrame, name: str, kwargs: dict):
    acc = set(accepted_kwargs())
    used = {k: v for k, v in kwargs.items() if k in acc}
    ignored = {k: v for k, v in kwargs.items() if k not in acc}

    print(f"\n=== {name} ===")
    print("Using kwargs:", json.dumps(used, ensure_ascii=False))
    if ignored:
        print("IGNORED by engine:", sorted(ignored.keys()))

    res = run_backtest(df, **used)
    summary, equity, trades = {}, None, None
    if isinstance(res, tuple) and len(res) == 3:
        summary, equity, trades = res
    elif isinstance(res, dict):
        summary = res
        trades = res.get("trades")
    else:
        summary = {}
    print("Summary:", json.dumps(summary or {}, ensure_ascii=False))
    try:
        tdf = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades)
    except Exception:
        tdf = pd.DataFrame()
    print("Trades:", 0 if tdf is None else len(tdf))
    if tdf is not None and not tdf.empty:
        cols = [c for c in ("EntryTime","EntryPrice","ExitTime","ExitPrice","PnL","reason") if c in tdf.columns]
        print("Trades head:\n", tdf[cols].head().to_string(index=False) if cols else tdf.head().to_string(index=False))

def main():
    ticker = "VOLV B"
    start, end = "2020-01-01", "2025-09-16"
    raw = get_ohlcv(ticker, start=start, end=end, source="borsdata")
    df = clean(pd.DataFrame(raw))
    print(f"Rows: {len(df)}  Period: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")

    # sanity Buy&Hold
    bh = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1.0) * 100.0
    print(f"Buy&Hold (sanity): {bh:.2f}%")

    # Case A: your known-good baseline
    base = dict(
        strategy="rsi",
        use_rsi_filter=True, rsi_window=14, rsi_min=25.0, rsi_max=60.0,
        breakout_lookback=55, exit_lookback=20,
        fast=15, slow=100,
        cost_bps=0.0, slip_bps=0
    )
    try_case(df, "KNOWN_BASELINE_RSI_PLUS_MAEXIT", base)

    # Case B: looser RSI (should create trades even if market is trending)
    loose = dict(
        strategy="rsi",
        use_rsi_filter=True, rsi_window=14, rsi_min=20.0, rsi_max=70.0,
        breakout_lookback=20, exit_lookback=10,
        fast=15, slow=100,
        cost_bps=0.0, slip_bps=0
    )
    try_case(df, "LOOSE_RSI", loose)

    # Case C: disable RSI gate entirely (to test engine path + MA Exit only)
    ma_only = dict(
        strategy="rsi",
        use_rsi_filter=False, rsi_window=14, rsi_min=25.0, rsi_max=60.0,
        breakout_lookback=0, exit_lookback=10,
        fast=15, slow=100,
        cost_bps=0.0, slip_bps=0
    )
    try_case(df, "MA_EXIT_ONLY", ma_only)

if __name__ == "__main__":
    main()
