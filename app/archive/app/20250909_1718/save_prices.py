# app/save_prices.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from app.backtest_simple import load_ohlcv

def save_borsdata_csv(ticker: str, start: str, end: str, out: str | None = None) -> Path:
    load_dotenv()
    df = load_ohlcv("borsdata", ticker, start, end)
    if df is None or df.empty:
        raise RuntimeError(f"Inga priser för {ticker} {start}..{end}")

    df = df.copy()
    df.index.name = "Date"
    # se till att standardkolumner finns i ordning
    cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    df = df[cols]

    outdir = Path("outputs/prices")
    outdir.mkdir(parents=True, exist_ok=True)
    if out:
        out_fp = Path(out)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
    else:
        safe = ticker.replace(" ", "_").replace("/", "-")
        out_fp = outdir / f"{safe}_{start}_{end}.csv"

    df.to_csv(out_fp)
    return out_fp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help='Ex. "VOLV B"')
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD")
    ap.add_argument("--out",   default=None, help="Valfri sökväg för CSV")
    args = ap.parse_args()

    out_fp = save_borsdata_csv(args.ticker, args.start, args.end, args.out)
    print(f"[OK] Sparat -> {out_fp.resolve()}")

if __name__ == "__main__":
    main()


