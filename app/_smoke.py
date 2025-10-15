from __future__ import annotations
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np

from app.data_providers import get_ohlcv
from app.backtest import run_backtest

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    need = ["Date","Open","High","Low","Close","Volume"]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Mappa datumkolumn till 'Date' om den heter något annat
    for a in ["Date","date","DATE","Day","day","time","Time","timestamp","Timestamp"]:
        if a in df.columns:
            if a != "Date":
                df = df.rename(columns={a:"Date"})
            break
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index":"Date"})

    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Saknar kolumner {missing}. Har: {list(df.columns)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need).sort_values("Date").reset_index(drop=True)
    return df

def rsi_baseline() -> Dict[str, Any]:
    return {
        "strategy": "rsi",
        "use_rsi_filter": True,
        "rsi_window": 14,
        "rsi_min": 25.0,
        "rsi_max": 60.0,
        "breakout_lookback": 55,
        "exit_lookback": 20,
        "use_trend_filter": False,
        "trend_ma_type": "EMA",
        "trend_ma_window": 100,
        "cost_bps": 0.0,
        "slip_bps": 0.0,
        "cash_rate_apy": 0.0,
        "max_positions": 1,
        "per_trade_pct": 100.0,
        "max_exposure_pct": 100.0,
    }

def parse_result(res: Any) -> Tuple[Dict[str, Any], pd.DataFrame | None, Any]:
    """
    Anpassad till din motor:
    - tuple på 3 element: (summary_dict, equity_any, trades_df)
    - annars: försök lista ut bästa tolkningen
    Returnerar: (summary_dict, trades_df | None, equity_any)
    """
    summary: Dict[str, Any] = {}
    trades_df: pd.DataFrame | None = None
    equity_any: Any = None

    if isinstance(res, tuple) and len(res) == 3:
        a, b, c = res
        if isinstance(a, dict):
            summary = a
        equity_any = b
        if isinstance(c, pd.DataFrame):
            trades_df = c
        elif isinstance(c, (list, tuple, dict)):
            try:
                trades_df = pd.DataFrame(c)
            except Exception:
                trades_df = None
        return summary, trades_df, equity_any

    # fallback: om dict direkt
    if isinstance(res, dict):
        summary = res
        # försök hitta trades i några vanliga keys
        for k in ("trades","Trades"):
            if k in res:
                try:
                    trades_df = res[k] if isinstance(res[k], pd.DataFrame) else pd.DataFrame(res[k])
                except Exception:
                    trades_df = None
                break
        return summary, trades_df, None

    # sista utväg: försök hitta en dict i tuplen/listan
    if isinstance(res, (list, tuple)):
        for x in res:
            if isinstance(x, dict):
                summary = x
            if trades_df is None:
                if isinstance(x, pd.DataFrame):
                    trades_df = x
                elif isinstance(x, (list, tuple, dict)):
                    try:
                        trades_df = pd.DataFrame(x)
                    except Exception:
                        pass
        return summary, trades_df, None

    return summary, trades_df, None

def buy_hold_return(df: pd.DataFrame) -> float:
    c0 = float(df["Close"].iloc[0]); c1 = float(df["Close"].iloc[-1])
    return (c1 / c0) - 1.0

def main():
    tkr = "VOLV B"
    start, end = "2020-01-01", "2025-09-16"
    raw = get_ohlcv(tkr, start=start, end=end, source="borsdata")
    df = clean_df(pd.DataFrame(raw))
    print(f"Rows: {len(df)}  Period: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")
    print(f"Buy&Hold (sanity): {buy_hold_return(df)*100:.2f}%")

    params = rsi_baseline()
    raw_res = run_backtest(df, **params)

    summary, trades_df, equity_any = parse_result(raw_res)
    print("Summary keys:", list(summary.keys()))
    print("Summary:", summary)

    if trades_df is None:
        print("Trades: (saknas eller okänt format)")
    else:
        print("Trades rows:", len(trades_df))
        cols = [c for c in ("EntryTime","EntryPrice","ExitTime","ExitPrice","PnL","reason") if c in trades_df.columns]
        if cols:
            print(trades_df[cols].head(5).to_string(index=False))
        else:
            print(trades_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
