#!/usr/bin/env python
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from multiprocessing import get_context

from app.data_providers import get_ohlcv
from app.opt_core import evaluate_candidate, pack_profile

# ---------- parametrisering ----------
def sample_params(rng: random.Random) -> Dict[str, Any]:
    use_rsi   = rng.random() < 0.8
    use_macd  = rng.random() < 0.4
    use_trend = rng.random() < 0.3
    use_bb    = rng.random() < 0.3
    use_sl    = rng.random() < 0.5
    use_atr_t = rng.random() < 0.5
    return {
        "use_rsi_filter": use_rsi,
        "rsi_window": rng.randint(8, 35),
        "rsi_min": round(rng.uniform(15, 35), 3),
        "rsi_max": round(rng.uniform(60, 80), 3),

        "use_trend_filter": use_trend,
        "trend_ma_type": rng.choice(["SMA","EMA"]),
        "trend_ma_window": rng.randint(50, 220),

        "breakout_lookback": rng.randint(10, 60),
        "exit_lookback": rng.randint(8, 55),

        "use_macd_filter": use_macd,
        "macd_fast": rng.randint(8, 18),
        "macd_slow": rng.randint(16, 30),
        "macd_signal": rng.randint(6, 14),

        "use_bb_filter": use_bb,
        "bb_window": rng.randint(12, 35),
        "bb_nstd": round(rng.uniform(1.5, 2.8), 3),
        "bb_min": round(rng.uniform(0.0, 0.8), 3),

        "use_stop_loss": use_sl,
        "stop_mode": "pct",
        "stop_loss_pct": round(rng.uniform(0.05, 0.2), 3),

        "atr_window": rng.randint(10, 22),
        "atr_mult": round(rng.uniform(1.3, 3.2), 3),
        "use_atr_trailing": use_atr_t,
        "atr_trail_mult": round(rng.uniform(1.0, 3.0), 3),

        "initial_equity": 100_000,
    }

def score_conservative(m: Dict[str, Any]) -> float:
    tr = float(m.get("TotalReturn", float("-inf")))
    dd = abs(float(m.get("MaxDD", 0.0) or 0.0))
    return tr - 1.2 * dd

def score_balanced(m: Dict[str, Any]) -> float:
    tr = float(m.get("TotalReturn", float("-inf")))
    dd = abs(float(m.get("MaxDD", 0.0) or 0.0))
    return tr - 0.7 * dd

def score_aggressive(m: Dict[str, Any]) -> float:
    tr = float(m.get("TotalReturn", float("-inf")))
    dd = abs(float(m.get("MaxDD", 0.0) or 0.0))
    return tr - 0.2 * dd

# ---------- multiprocessing worker ----------
_DF: Optional[pd.DataFrame] = None
def _init_worker(df: pd.DataFrame):
    global _DF
    _DF = df

def _worker(params: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        m = evaluate_candidate(_DF, params, return_series=False)
        return {"params": params, "metrics": m}, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ---------- helpers ----------
def _better(best: Optional[Tuple[Dict[str,Any],Dict[str,Any]]],
           cand: Tuple[Dict[str,Any],Dict[str,Any]],
           scorer) -> bool:
    if best is None:
        return True
    return scorer(cand[1]) > scorer(best[1])

def _save_checkpoint(path: Path, ticker: str,
                     cons: Optional[Tuple[Dict[str,Any],Dict[str,Any]]],
                     bal:  Optional[Tuple[Dict[str,Any],Dict[str,Any]]],
                     aggr: Optional[Tuple[Dict[str,Any],Dict[str,Any]]],
                     done: int, total: int):
    def pack(x, name):
        if x is None:
            return None
        return pack_profile(ticker, name, x[0], x[1])
    payload = {
        "progress": {"done": done, "total": total, "ts": int(time.time())},
        "profiles": [p for p in (
            pack(cons, "conservative"),
            pack(bal, "balanced"),
            pack(aggr, "aggressive"),
        ) if p is not None]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    pa = argparse.ArgumentParser(description="Parallell optimizer som skriver 3 profiler till profiles/<TICKER>.json")
    pa.add_argument("ticker")
    pa.add_argument("--start", default="2020-10-04")
    pa.add_argument("--end", default=None)
    pa.add_argument("--samples", type=int, default=20000)
    pa.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--checkpoint-every", type=int, default=1000)
    pa.add_argument("--resume", action="store_true", help="om det finns .progress, använd det som start-best")
    args = pa.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Hämta DF en gång (Börsdata)
    df = get_ohlcv(args.ticker, start=args.start, end=args.end)
    if df is None or len(df) < 50:
        raise SystemExit(f"Kunde inte hämta data för {args.ticker} ({args.start}→{args.end}).")

    # Progress/utdata paths
    out_dir = Path("profiles"); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.ticker}.json"
    prog_json = out_dir / f"{args.ticker}.progress.json"

    cons: Optional[Tuple[Dict[str,Any],Dict[str,Any]]] = None
    bal:  Optional[Tuple[Dict[str,Any],Dict[str,Any]]] = None
    aggr: Optional[Tuple[Dict[str,Any],Dict[str,Any]]] = None
    done = 0

    if args.resume and prog_json.exists():
        try:
            prog = json.loads(prog_json.read_text(encoding="utf-8"))
            profs = {p["name"].split(" – ")[-1]: p for p in prog.get("profiles", [])}
            if "conservative" in profs: cons = (profs["conservative"]["params"], profs["conservative"]["metrics"])
            if "balanced"     in profs: bal  = (profs["balanced"]["params"],     profs["balanced"]["metrics"])
            if "aggressive"   in profs: aggr = (profs["aggressive"]["params"],   profs["aggressive"]["metrics"])
            done = int(prog.get("progress", {}).get("done", 0))
            print(f"[resume] Återupptar från {prog_json} (done={done})")
        except Exception:
            pass

    # Generator av param-kandidater
    def param_iter():
        i = 0
        # Fortsätt där vi var om resume
        while i < args.samples:
            p = sample_params(rng)
            p["from_date"] = args.start
            if args.end: p["to_date"] = args.end
            i += 1
            yield p

    # Kör parallellt
    ctx = get_context("fork")  # effektivare på Linux
    with ctx.Pool(processes=max(1, int(args.jobs)), initializer=_init_worker, initargs=(df,)) as pool:
        for res, err in pool.imap_unordered(_worker, param_iter(), chunksize=64):
            done += 1
            if res is not None:
                params, metrics = res["params"], res["metrics"]
                cand = (params, metrics)
                if _better(cons, cand, score_conservative): cons = cand
                if _better(bal,  cand, score_balanced):     bal  = cand
                if _better(aggr, cand, score_aggressive):   aggr = cand
            if args.checkpoint_every and done % args.checkpoint_every == 0:
                _save_checkpoint(prog_json, args.ticker, cons, bal, aggr, done, args.samples)
                print(f"[ckpt] {done}/{args.samples} sparat → {prog_json}")

    # Skriv slutligt JSON (ersätt progress)
    final_profiles = [
        pack_profile(args.ticker, "conservative", cons[0], cons[1]) if cons else None,
        pack_profile(args.ticker, "balanced",     bal[0],  bal[1])  if bal  else None,
        pack_profile(args.ticker, "aggressive",   aggr[0], aggr[1]) if aggr else None,
    ]
    final_profiles = [p for p in final_profiles if p is not None]
    if not final_profiles:
        raise SystemExit("Inga giltiga profiler hittades.")

    out = {"profiles": final_profiles}
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if prog_json.exists():
        try: prog_json.unlink()
        except Exception: pass

    print(f"[OK] Sparade: {out_json}")
    tr = [round(p["metrics"]["TotalReturn"], 4) for p in final_profiles]
    bh = [round(p["metrics"]["BuyHold"], 4)     for p in final_profiles]
    dd = [round(p["metrics"].get("MaxDD", float('nan')), 4) for p in final_profiles]
    print("  TR (cons/bal/agg):", *tr)
    print("  BH (cons/bal/agg):", *bh)
    print("  DD (cons/bal/agg):", *dd)
