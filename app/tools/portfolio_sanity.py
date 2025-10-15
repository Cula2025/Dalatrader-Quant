from __future__ import annotations
import os, sys
import pandas as pd
from typing import List
from app import btwrap as W
from app.data_providers import get_ohlcv
from app.equity_extract import extract_equity
from app.portfolio_math import equal_weight_rebalanced  # vi använder bara rebalanced här

INDEX_TICKER = "OMXS30GI"

def pick_first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def get_prices(ticker: str, start: str) -> pd.DataFrame:
    df = get_ohlcv(ticker=ticker, start=start, end=None)[["Close"]].dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

def eq_for(ticker: str, start: str, price_index: pd.DatetimeIndex) -> pd.Series:
    """Kör backtest → equity, och aligna equity-datum till prisindex (tail-align)."""
    res = W.run_backtest(p={"ticker": ticker, "params": {"from_date": start}})
    x   = pick_first(res.get("equity"), res.get("summary"), res)
    s   = extract_equity(x)
    s   = pd.to_numeric(s, errors="coerce").dropna()

    # Tail-align: justera längder och lägg equitys index till prisernas datum
    if len(s) > len(price_index):
        s = s.iloc[-len(price_index):]
    elif len(s) < len(price_index):
        price_index = price_index[-len(s):]
    s.index = price_index
    return s.sort_index()

def buyhold_from_prices(price_frames: List[pd.DataFrame], tickers: List[str], common: pd.DatetimeIndex) -> pd.Series:
    # Bygg en tabell P med Close-kolumner, strikt på 'common'
    P = pd.concat([df.reindex(common)["Close"].rename(t) for df, t in zip(price_frames, tickers)], axis=1).dropna()
    R = P / P.iloc[0]        # prisrelativ från start
    bh = R.mean(axis=1)      # equal-weight Buy&Hold
    bh.name = "Buy&Hold"
    return bh

def main(argv):
    start = os.getenv("START", "2020-10-04")
    tickers = [a for a in argv if not a.startswith("--")]
    if not tickers:
        tickers = ["BOL", "EQT", "epi-a"]

    print("START:", start)
    print("TICKERS:", ", ".join(tickers))

    # 1) Hämta priser först och bygg common-datum från dem
    price_frames = [get_prices(t, start) for t in tickers]
    common = price_frames[0].index
    for df in price_frames[1:]:
        common = common.intersection(df.index)
    common = common.sort_values()
    if len(common) == 0:
        print("COMMON=0 → avbryter.")
        return 1

    # 2) Strategi-equities, alignade till respektive prisindex, sedan reindex till common
    eqs = []
    for t, dfp in zip(tickers, price_frames):
        s = eq_for(t, start, dfp.index)
        s = s.reindex(common).ffill().bfill()
        eqs.append(s)

    # 3) Portfölj (EW, rebalanced) på common
    port = equal_weight_rebalanced(eqs).reindex(common).ffill().bfill()
    port.name = "Portfolio"

    # 4) Buy&Hold (EW av pris)
    bh = buyhold_from_prices(price_frames, tickers, common)

    # 5) Index (OMXS30GI) på common
    idx = get_ohlcv(INDEX_TICKER, start=start, end=None)["Close"].dropna()
    if not isinstance(idx.index, pd.DatetimeIndex):
        idx.index = pd.to_datetime(idx.index)
    idx = idx.sort_index().reindex(common).ffill().bfill()
    idx_n = (idx / idx.iloc[0]).rename(INDEX_TICKER)

    # 6) Utskrift + CSV
    print(f"INTERVALL: {common[0].date()} -> {common[-1].date()}  (n={len(common)})")
    print(f"Portfolio_final   ≈ {float(port.iloc[-1]):.4f}x")
    print(f"Buy&Hold_final    ≈ {float(bh.iloc[-1]):.4f}x")
    print(f"{INDEX_TICKER}_final  ≈ {float(idx_n.iloc[-1]):.4f}x")

    out = pd.concat([port, bh, idx_n], axis=1)
    out.to_csv("profiles/_sanity_curves.csv")
    print("Wrote profiles/_sanity_curves.csv")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
