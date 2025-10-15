from __future__ import annotations
import json, math, itertools
from dataclasses import asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

# --- Platser ---
ROOT = Path(__file__).resolve().parent
PRICES_DIR = ROOT / "outputs" / "prices"
PROFILES_DIR = ROOT / "outputs" / "profiles"
OPT_OUT_DIR = ROOT / "outputs" / "opt_results"
for d in (PROFILES_DIR, OPT_OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Din backtester ---
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.portfolio_backtest as PB
from app.portfolio_backtest import PortfolioParams

TICKERS = ["SHB A", "ERIC B", "INVE B", "VOLV B"]  # ABB är redan klar hos dig, men kan läggas till här om du vill
START = date(2020, 1, 1)
END: Optional[date] = None  # None = till senaste i CSV

# --- Hjälp ---
def _norm(t: str) -> str:
    return t.replace(" ", "_").replace("/", "-")

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns:
            df[c] = 0.0 if c == "Volume" else df["Close"]
    return df[["Open","High","Low","Close","Volume"]]

def load_csv(ticker: str, start: Optional[date], end: Optional[date]) -> Optional[pd.DataFrame]:
    cands = sorted(PRICES_DIR.glob(f"*{_norm(ticker)}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    df = pd.read_csv(cands[0])
    df = _ensure_cols(df)
    if start:
        df = df[df.index.date >= start]
    if end:
        df = df[df.index.date <= end]
    return df if not df.empty else None

def buyhold(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return float("nan")
    c0, c1 = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
    if c0 <= 0:
        return float("nan")
    return (c1 / c0) - 1.0

# --- Grid (slimmad & vettig) ---
def build_space() -> List[Dict[str, Any]]:
    space: List[Dict[str, Any]] = []

    # MA Cross
    ma_fast = [5, 8, 10, 12, 15, 20]
    ma_slow = [50, 80, 100, 150, 200]
    trend = [0, 150, 200]
    for f, s, tw in itertools.product(ma_fast, ma_slow, trend):
        if f >= s:
            continue
        space.append(dict(strategy="ma_cross", trend_ma_window=tw, fast=f, slow=s))

    # RSI
    rsi_win = [7, 10, 14]
    rsi_min = [25, 30, 35, 40]
    rsi_max = [60, 65, 70, 75]
    for rw, rmin, rmax, tw in itertools.product(rsi_win, rsi_min, rsi_max, trend):
        if rmin >= rmax:
            continue
        space.append(dict(strategy="rsi", trend_ma_window=tw, rsi_window=rw, rsi_min=float(rmin), rsi_max=float(rmax)))

    # Breakout (Donchian)
    bo_lb = [20, 55, 100]
    bo_exit = [10, 20, 50]
    for lb, ex, tw in itertools.product(bo_lb, bo_exit, trend):
        space.append(dict(strategy="breakout", trend_ma_window=tw, breakout_lookback=lb, exit_lookback=ex))

    return space

def make_params(b: Dict[str, Any]) -> PortfolioParams:
    return PortfolioParams(
        strategy=b.get("strategy", "ma_cross"),
        use_trend_filter=bool(b.get("trend_ma_window", 0) > 0),
        trend_ma_type="EMA",
        trend_ma_window=int(b.get("trend_ma_window", 0)),
        fast=int(b.get("fast", 15)),
        slow=int(b.get("slow", 100)),
        use_rsi_filter=b.get("strategy") == "rsi",
        rsi_window=int(b.get("rsi_window", 14)),
        rsi_min=float(b.get("rsi_min", 30.0)),
        rsi_max=float(b.get("rsi_max", 70.0)),
        breakout_lookback=int(b.get("breakout_lookback", 55)),
        exit_lookback=int(b.get("exit_lookback", 20)),
        # allt annat av / default
        use_macd_filter=False, macd_fast=12, macd_slow=26, macd_signal=9, macd_mode="above_zero",
        use_bb_filter=False, bb_window=20, bb_nstd=2.0, bb_mode="exit_below_mid", bb_percent_b_min=0.8,
        atr_window=14, atr_stop_mult=0.0, atr_trail_mult=0.0,
        cost_bps=0.0, cash_rate_apy=0.0,
        max_positions=1, per_trade_pct=100.0, max_exposure_pct=100.0,
    )

def run_one(ticker: str, df: pd.DataFrame, base: Dict[str, Any]) -> Dict[str, Any]:
    p = make_params(base)
    equity, trades, stats = PB.run_portfolio_backtest({ticker: df}, p)
    return {
        "Ticker": ticker,
        "Strategy": p.strategy,
        "Params": base,
        "TotalReturn": float(stats.get("TotalReturn", float("nan"))),
        "MaxDD": float(stats.get("MaxDD", float("nan"))),
        "SharpeD": float(stats.get("SharpeD", float("nan"))),
        "LengthDays": int(stats.get("LengthDays", 0)),
        "__version__": stats.get("__version__", "?"),
    }

def choose_best(rows: List[Dict[str, Any]], bh: float) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: (r.get("TotalReturn", float("-inf")), r.get("SharpeD", float("-inf"))), reverse=True)
    for r in rows:
        if not math.isnan(bh) and not math.isnan(r["TotalReturn"]) and r["TotalReturn"] > bh:
            return r
    return rows[0]

def save_all(ticker: str, rows: List[Dict[str, Any]], best: Dict[str, Any], bh: float) -> Tuple[Path, Path]:
    df = pd.DataFrame(rows)
    res_fp = OPT_OUT_DIR / f"{_norm(ticker)}_results.csv"
    df.to_csv(res_fp, index=False, encoding="utf-8-sig")

    params = make_params(best["Params"])
    payload = {
        "profiles": [
            {
                "name": f"{ticker} – auto_best",
                "ticker": ticker,
                "params": asdict(params),
                "metrics": {
                    "TotalReturn": best["TotalReturn"],
                    "MaxDD": best["MaxDD"],
                    "SharpeD": best["SharpeD"],
                    "BuyHold": bh,
                },
            }
        ]
    }
    prof_fp = PROFILES_DIR / f"{_norm(ticker)}_best.json"
    prof_fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return res_fp, prof_fp

def optimize_ticker(ticker: str):
    print(f"\n=== {ticker} ===")
    df = load_csv(ticker, START, END)
    if df is None:
        print("❌ Hittar ingen CSV i outputs/prices/")
        return

    bh = buyhold(df)
    print(f"Buy&Hold: {bh*100:,.2f}%  | rader: {len(df)}")
    space = build_space()

    rows: List[Dict[str, Any]] = []
    for base in space:
        rows.append(run_one(ticker, df, base))

    best = choose_best(rows, bh)
    if not best:
        print("❌ Ingen vinnare hittad.")
        return
    res_fp, prof_fp = save_all(ticker, rows, best, bh)
    print(f"✅ Bästa: {best['Strategy']}  TR={best['TotalReturn']*100:,.2f}%, SharpeD={best['SharpeD']:.2f}")
    print(f"   Resultat: {res_fp}")
    print(f"   Profil:   {prof_fp}")

def main():
    for t in TICKERS:
        optimize_ticker(t)

if __name__ == "__main__":
    main()

