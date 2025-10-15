from __future__ import annotations
import json, math, itertools, os, sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures as cf

import pandas as pd

# --- Projektroot & importer ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.portfolio_backtest as PB
from app.portfolio_backtest import PortfolioParams

PRICES_DIR   = ROOT / "outputs" / "prices"
PROFILES_DIR = ROOT / "outputs" / "profiles"
OPT_OUT_DIR  = ROOT / "outputs" / "opt_results"
for d in (PRICES_DIR, PROFILES_DIR, OPT_OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ===== Helpers =====
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

# ===== Search space (~1000+ combos/ticker) =====
def build_space() -> List[Dict[str, Any]]:
    space: List[Dict[str, Any]] = []

    # Trendfilter (EMA)
    trend_windows = [0, 100, 150, 200]

    # MACD-gate
    macd_use     = [False, True]
    macd_triples = [(12,26,9), (8,17,9)]
    macd_modes   = ["above_zero"]

    # Bollinger-gate
    bb_use     = [False, True]
    bb_windows = [20]
    bb_nstds   = [2.0]
    bb_modes   = ["exit_below_mid"]
    bb_pctbmin = [0.8]

    # 1) MA Cross
    ma_fast = [5, 8, 10, 12, 15, 20]
    ma_slow = [50, 80, 100, 150, 200]
    for f, s, tw, use_macd, use_bb in itertools.product(ma_fast, ma_slow, trend_windows, macd_use, bb_use):
        if f >= s:
            continue
        base = dict(strategy="ma_cross",
                    trend_ma_window=tw, trend_ma_type="EMA",
                    fast=f, slow=s)
        base.update({"use_macd_filter": use_macd, "use_bb_filter": use_bb})
        if use_macd:
            for (mf, ms, sig), mode in itertools.product(macd_triples, macd_modes):
                b2 = dict(base, macd_fast=mf, macd_slow=ms, macd_signal=sig, macd_mode=mode)
                if use_bb:
                    for bw, bn, bm, pb in itertools.product(bb_windows, bb_nstds, bb_modes, bb_pctbmin):
                        b3 = dict(b2, bb_window=bw, bb_nstd=bn, bb_mode=bm, bb_percent_b_min=pb)
                        space.append(b3)
                else:
                    space.append(b2)
        else:
            if use_bb:
                for bw, bn, bm, pb in itertools.product(bb_windows, bb_nstds, bb_modes, bb_pctbmin):
                    b2 = dict(base, bb_window=bw, bb_nstd=bn, bb_mode=bm, bb_percent_b_min=pb)
                    space.append(b2)
            else:
                space.append(base)

    # 2) RSI kanal
    rsi_win = [7, 10, 14]
    rsi_min = [25, 30, 35, 40]
    rsi_max = [60, 65, 70, 75]
    for rw, rmin, rmax, tw, use_macd, use_bb in itertools.product(rsi_win, rsi_min, rsi_max, trend_windows, macd_use, bb_use):
        if rmin >= rmax:
            continue
        base = dict(strategy="rsi",
                    rsi_window=rw, rsi_min=float(rmin), rsi_max=float(rmax),
                    trend_ma_window=tw, trend_ma_type="EMA")
        base.update({"use_macd_filter": use_macd, "use_bb_filter": use_bb})
        if use_macd:
            for (mf, ms, sig), mode in itertools.product(macd_triples, macd_modes):
                b2 = dict(base, macd_fast=mf, macd_slow=ms, macd_signal=sig, macd_mode=mode)
                if use_bb:
                    for bw, bn, bm, pb in itertools.product(bb_windows, bb_nstds, bb_modes, bb_pctbmin):
                        b3 = dict(b2, bb_window=bw, bb_nstd=bn, bb_mode=bm, bb_percent_b_min=pb)
                        space.append(b3)
                else:
                    space.append(b2)
        else:
            if use_bb:
                for bw, bn, bm, pb in itertools.product(bb_windows, bb_nstds, bb_modes, bb_pctbmin):
                    b2 = dict(base, bb_window=bw, bb_nstd=bn, bb_mode=bm, bb_percent_b_min=pb)
                    space.append(b2)
            else:
                space.append(base)

    # 3) Breakout (Donchian)
    bo_lb   = [20, 55, 100]
    bo_exit = [10, 20, 50]
    for lb, ex, tw, use_macd, use_bb in itertools.product(bo_lb, bo_exit, trend_windows, macd_use, bb_use):
        base = dict(strategy="breakout",
                    breakout_lookback=lb, exit_lookback=ex,
                    trend_ma_window=tw, trend_ma_type="EMA")
        base.update({"use_macd_filter": use_macd, "use_bb_filter": use_bb})
        if use_macd:
            for (mf, ms, sig), mode in itertools.product(macd_triples, macd_modes):
                b2 = dict(base, macd_fast=mf, macd_slow=ms, macd_signal=sig, macd_mode=mode)
                if use_bb:
                    for bw, bn, bm, pb in itertools.product(bb_windows, bb_nstds, bb_modes, bb_pctbmin):
                        b3 = dict(b2, bb_window=bw, bb_nstd=bn, bb_mode=bm, bb_percent_b_min=pb)
                        space.append(b3)
                else:
                    space.append(b2)
        else:
            if use_bb:
                for bw, bn, bm, pb in itertools.product(bb_windows, bb_nstds, bb_modes, bb_pctbmin):
                    b2 = dict(base, bb_window=bw, bb_nstd=bn, bb_mode=bm, bb_percent_b_min=pb)
                    space.append(b2)
            else:
                space.append(base)

    # Gallra lite om allt blev enormt
    MAX_COMBOS = 1600
    if len(space) > MAX_COMBOS:
        step = max(1, len(space)//MAX_COMBOS)
        space = space[::step]
    return space

def make_params(b: Dict[str, Any]) -> PortfolioParams:
    return PortfolioParams(
        strategy=b.get("strategy", "ma_cross"),
        # trend
        use_trend_filter=bool(b.get("trend_ma_window", 0) > 0),
        trend_ma_type=b.get("trend_ma_type", "EMA"),
        trend_ma_window=int(b.get("trend_ma_window", 0)),
        # MA
        fast=int(b.get("fast", 15)),
        slow=int(b.get("slow", 100)),
        # RSI
        use_rsi_filter=b.get("strategy") == "rsi",
        rsi_window=int(b.get("rsi_window", 14)),
        rsi_min=float(b.get("rsi_min", 30.0)),
        rsi_max=float(b.get("rsi_max", 70.0)),
        # Breakout
        breakout_lookback=int(b.get("breakout_lookback", 55)),
        exit_lookback=int(b.get("exit_lookback", 20)),
        # MACD gate
        use_macd_filter=bool(b.get("use_macd_filter", False)),
        macd_fast=int(b.get("macd_fast", 12)),
        macd_slow=int(b.get("macd_slow", 26)),
        macd_signal=int(b.get("macd_signal", 9)),
        macd_mode=b.get("macd_mode", "above_zero"),
        # Bollinger gate
        use_bb_filter=bool(b.get("use_bb_filter", False)),
        bb_window=int(b.get("bb_window", 20)),
        bb_nstd=float(b.get("bb_nstd", 2.0)),
        bb_mode=b.get("bb_mode", "exit_below_mid"),
        bb_percent_b_min=float(b.get("bb_percent_b_min", 0.8)),
        # Risk & kapital
        atr_window=14, atr_stop_mult=0.0, atr_trail_mult=0.0,
        cost_bps=0.0, cash_rate_apy=0.0,
        max_positions=1, per_trade_pct=100.0, max_exposure_pct=100.0,
    )

def run_combo(ticker: str, df: pd.DataFrame, base: Dict[str, Any]) -> Dict[str, Any]:
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

# ---- Tråd-worker (pickle-säker) ----
def _worker(args: Tuple[str, pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
    ticker, df, base = args
    return run_combo(ticker, df, base)

def choose_best(rows: List[Dict[str, Any]], bh: float, metric: str = "TotalReturn") -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    rows_sorted = sorted(rows, key=lambda r: (r.get(metric, float("-inf")), r.get("SharpeD", float("-inf"))), reverse=True)
    for r in rows_sorted:
        if not math.isnan(bh) and not math.isnan(r["TotalReturn"]) and r["TotalReturn"] > bh:
            return r
    return rows_sorted[0]

def save_outputs(ticker: str, rows: List[Dict[str, Any]], best: Dict[str, Any], bh: float) -> Tuple[Path, Path]:
    res_df = pd.DataFrame(rows)
    res_fp = OPT_OUT_DIR / f"{_norm(ticker)}_results.csv"
    res_df.to_csv(res_fp, index=False, encoding="utf-8-sig")

    params = make_params(best["Params"])
    payload = {
        "profiles": [{
            "name": f"{ticker} – auto_best_1000",
            "ticker": ticker,
            "params": asdict(params),
            "metrics": {
                "TotalReturn": best["TotalReturn"],
                "MaxDD": best["MaxDD"],
                "SharpeD": best["SharpeD"],
                "BuyHold": bh,
            },
        }]
    }
    prof_fp = PROFILES_DIR / f"{_norm(ticker)}_best.json"
    prof_fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return res_fp, prof_fp

def optimize_ticker(ticker: str, start: date, end: Optional[date], max_workers: int = 4):
    print(f"\n=== {ticker} ===")
    df = load_csv(ticker, start, end)
    if df is None:
        print("❌ Ingen CSV hittad i outputs/prices/")
        return
    bh = buyhold(df)
    print(f"Buy&Hold: {bh*100:,.2f}%  | rader: {len(df)}")

    space = build_space()
    print(f"Kör kombinationer: {len(space)}")

    rows: List[Dict[str, Any]] = []

    # ---- Kör med trådar (pickle-fritt) ----
    tasks = ((ticker, df, b) for b in space)
    try:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, r in enumerate(ex.map(_worker, tasks, timeout=None), 1):
                rows.append(r)
                if i % 200 == 0:
                    print(f"  … {i}/{len(space)} klart")
    except Exception as e:
        # Fallback single-thread om något miljöproblem uppstår
        print(f"⚠️ Tråd-exekvering föll tillbaka till single-thread: {e}")
        rows = []
        for i, b in enumerate(space, 1):
            rows.append(run_combo(ticker, df, b))
            if i % 200 == 0:
                print(f"  … {i}/{len(space)} klart")

    best = choose_best(rows, bh, metric="TotalReturn")
    if not best:
        print("❌ Ingen vinnare hittad.")
        return
    res_fp, prof_fp = save_outputs(ticker, rows, best, bh)
    print(f"✅ Bästa: {best['Strategy']} | TR={best['TotalReturn']*100:,.2f}%, SharpeD={best['SharpeD']:.2f}")
    print(f"   Resultat: {res_fp}")
    print(f"   Profil:   {prof_fp}")

def main():
    # Justera listan här
    tickers = ["SHB A", "ERIC B", "INVE B", "VOLV B"]
    start = date(2020, 1, 1)
    end: Optional[date] = None  # till senaste
    # Rimligt antal trådar
    try:
        cpu = os.cpu_count() or 4
    except Exception:
        cpu = 4
    max_workers = min(16, max(4, cpu))  # lite mer konka för att få fart

    for t in tickers:
        optimize_ticker(t, start, end, max_workers=max_workers)

if __name__ == "__main__":
    main()



