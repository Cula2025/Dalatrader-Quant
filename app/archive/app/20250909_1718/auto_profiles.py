from __future__ import annotations

import argparse, json, itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Vi använder endast de publika funktionerna i backtest_simple
from app.backtest_simple import load_ohlcv, run_backtest


# ---------- Robust läsning av universum ----------
def _load_universe(path: Path) -> List[str]:
    if not path.exists():
        return ["ATCO A","INVE B","VOLV B","SEB A","ASSA B","SWED A","SAAB B","SAND","ABB","ERIC B"]
    text: Optional[str] = None
    for enc in ("utf-8-sig","utf-8","utf-16","latin-1"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = path.read_text()
    out: List[str] = []
    for line in text.splitlines():
        s = line.strip().lstrip("\ufeff")  # ta bort ev. BOM
        if s and not s.startswith("#"):
            out.append(s)
    return out


# ---------- Parametrar för grid ----------
def _param_grid(mode: str) -> Dict[str, List]:
    mode = (mode or "medium").lower()
    if mode == "small":
        return {
            "breakout_lookback": [15, 20],
            "exit_lookback":     [5, 10],
            "use_trend_filter":  [False, True],
            "trend_ma_window":   [100],
            "use_rsi_filter":    [False, True],
            "rsi_period":        [14],
            "rsi_enter_min":     [0.0, 50.0],      # 0.0 = av
            "rsi_exit_max":      [None, 70.0],
            "stop_loss_pct":     [None, 0.08],
            "take_profit_pct":   [None, 0.15],
            "trailing_stop_pct": [None, 0.10],
            "breakeven_atr_mult":[0.0, 1.0],
            "cooldown_days_after_sl": [0, 14],
        }
    if mode == "large":
        return {
            "breakout_lookback": [10, 15, 20, 30, 40],
            "exit_lookback":     [5, 8, 10, 15, 20],
            "use_trend_filter":  [False, True],
            "trend_ma_window":   [50, 100, 150, 200],
            "use_rsi_filter":    [False, True],
            "rsi_period":        [14],
            "rsi_enter_min":     [0.0, 45.0, 50.0, 55.0],
            "rsi_exit_max":      [None, 65.0, 70.0],
            "stop_loss_pct":     [None, 0.05, 0.08, 0.10],
            "take_profit_pct":   [None, 0.10, 0.15, 0.20],
            "trailing_stop_pct": [None, 0.08, 0.10, 0.12],
            "breakeven_atr_mult":[0.0, 0.5, 1.0],
            "cooldown_days_after_sl": [0, 10, 20],
        }
    # default: medium
    return {
        "breakout_lookback": [15, 20, 30],
        "exit_lookback":     [5, 10, 15],
        "use_trend_filter":  [False, True],
        "trend_ma_window":   [50, 100, 150],
        "use_rsi_filter":    [False, True],
        "rsi_period":        [14],
        "rsi_enter_min":     [0.0, 50.0],      # 0.0 = av
        "rsi_exit_max":      [None, 70.0],
        "stop_loss_pct":     [None, 0.05, 0.08],
        "take_profit_pct":   [None, 0.15, 0.20],
        "trailing_stop_pct": [None, 0.10],
        "breakeven_atr_mult":[0.0, 1.0],
        "cooldown_days_after_sl": [0, 14],
    }


def _iter_param_combos(grid: Dict[str, List]) -> List[Dict]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = []
    for prod in itertools.product(*vals):
        p = dict(zip(keys, prod))
        combos.append(p)
    return combos


def _risk_score(maxdd: float, pf: float) -> int:
    dd_score = 1 if maxdd > -0.10 else 2 if maxdd > -0.20 else 3 if maxdd > -0.30 else 4 if maxdd > -0.40 else 5
    pf_bonus = -1 if pf >= 2.0 else (-0.5 if pf >= 1.5 else 0)
    return int(min(5, max(1, dd_score + pf_bonus)))


def _tune_one(ticker: str, start: str, end: str, grid_mode: str) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    try:
        df = load_ohlcv("borsdata", ticker, start, end)
        if df is None or df.empty:
            return None, None
    except Exception as e:
        print(f"[WARN] {ticker}: datafel: {e}", flush=True)
        return None, None

    grid = _param_grid(grid_mode)
    combos = _iter_param_combos(grid)
    best: Optional[Dict] = None
    best_rank = (-1e9, -1e9, 1e9)  # sort: Sharpe desc, TotalReturn desc, MaxDD (mindre negativ) desc

    for i, p in enumerate(combos, 1):
        try:
            eq, tr, st = run_backtest(
                df,
                strategy="breakout",
                breakout_lookback=int(p["breakout_lookback"]),
                exit_lookback=int(p["exit_lookback"]),
                use_trend_filter=bool(p["use_trend_filter"]),
                trend_ma_window=int(p["trend_ma_window"]),
                use_rsi_filter=bool(p["use_rsi_filter"]),
                rsi_period=int(p["rsi_period"]),
                rsi_enter_min=float(p["rsi_enter_min"]),
                rsi_exit_max=(None if p["rsi_exit_max"] is None else float(p["rsi_exit_max"])),
                stop_loss_pct=(None if p["stop_loss_pct"] is None else float(p["stop_loss_pct"])),
                take_profit_pct=(None if p["take_profit_pct"] is None else float(p["take_profit_pct"])),
                trailing_stop_pct=(None if p["trailing_stop_pct"] is None else float(p["trailing_stop_pct"])),
                breakeven_atr_mult=float(p["breakeven_atr_mult"]),
                cooldown_days_after_sl=int(p["cooldown_days_after_sl"]),
            )
            sharpe = float(st.get("SharpeD", 0.0))
            tret   = float(st.get("TotalReturn", 0.0))
            maxdd  = float(st.get("MaxDD", 0.0))
            rank = (sharpe, tret, -maxdd)
            if rank > best_rank:
                best_rank = rank
                best = {"Ticker": ticker, "Params": p, "Stats": st}
        except Exception:
            continue

        if i % 200 == 0:
            print(f"[INFO] {ticker}: {i}/{len(combos)} körda…", flush=True)

    if best is None:
        return None, None

    st = best["Stats"]
    p  = best["Params"]
    row = {
        "Ticker": ticker,
        "Strategy": "breakout",
        "TotalReturn": float(st.get("TotalReturn", 0.0)),
        "MaxDD": float(st.get("MaxDD", 0.0)),
        "SharpeD": float(st.get("SharpeD", 0.0)),
        "Trades": int(st.get("Trades", 0)),
        "WinRate": float(st.get("WinRate", 0.0)),
        "PF": float(st.get("PF", 0.0)),
        "Risk": _risk_score(float(st.get("MaxDD", 0.0)), float(st.get("PF", 0.0))),
        "breakout_lookback": int(p["breakout_lookback"]),
        "exit_lookback": int(p["exit_lookback"]),
        "use_trend_filter": bool(p["use_trend_filter"]),
        "trend_ma_window": int(p["trend_ma_window"]),
        "use_rsi_filter": bool(p["use_rsi_filter"]),
        "rsi_period": int(p["rsi_period"]),
        "rsi_enter_min": float(p["rsi_enter_min"]),
        "rsi_exit_max": ("" if p["rsi_exit_max"] is None else float(p["rsi_exit_max"])),
        "stop_loss_pct": ("" if p["stop_loss_pct"] is None else float(p["stop_loss_pct"])),
        "take_profit_pct": ("" if p["take_profit_pct"] is None else float(p["take_profit_pct"])),
        "trailing_stop_pct": ("" if p["trailing_stop_pct"] is None else float(p["trailing_stop_pct"])),
        "breakeven_atr_mult": float(p["breakeven_atr_mult"]),
        "cooldown_days_after_sl": int(p["cooldown_days_after_sl"]),
    }
    return best, pd.DataFrame([row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, help="Textfil med en ticker per rad")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--mode", choices=["small","medium","large"], default="medium")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    tickers = _load_universe(Path(args.universe))
    print(f"[INFO] Startar autotune på {len(tickers)} tickers, {args.start}..{args.end}", flush=True)

    profiles: Dict[str, Dict] = {}
    rows: List[pd.DataFrame] = []

    for i, t in enumerate(tickers, 1):
        print(f"[INFO] ({i}/{len(tickers)}) {t} …", flush=True)
        best, row = _tune_one(t, args.start, args.end, args.mode)
        if best is None:
            print(f"[WARN] {t}: ingen giltig kombo hittad", flush=True)
            continue
        profiles[t] = best
        rows.append(row)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False)

    if rows:
        df = pd.concat(rows, ignore_index=True)
        df = df.sort_values(["SharpeD","TotalReturn","MaxDD"], ascending=[False, False, False])
        df.to_csv(args.out_csv, index=False, encoding="utf-8")
        print(f"[OK] Profiler -> {args.out_json}")
        print(f"[OK] Sammanställning -> {args.out_csv}")
    else:
        pd.DataFrame(columns=[
            "Ticker","Strategy","TotalReturn","MaxDD","SharpeD","Trades","WinRate","PF","Risk",
            "breakout_lookback","exit_lookback","use_trend_filter","trend_ma_window",
            "use_rsi_filter","rsi_period","rsi_enter_min","rsi_exit_max",
            "stop_loss_pct","take_profit_pct","trailing_stop_pct",
            "breakeven_atr_mult","cooldown_days_after_sl"
        ]).to_csv(args.out_csv, index=False, encoding="utf-8")
        print("[WARN] Inga resultat att skriva (alla tickers misslyckades?)")
        print(f"[OK] Profiler -> {args.out_json}")


if __name__ == "__main__":
    main()

