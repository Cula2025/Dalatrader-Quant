from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
from typing import Dict, Any, Iterable

import pandas as pd
from app import btwrap as W
from app.equity_extract import extract_equity

DEF_FROM_DATE = "2020-10-04"  # samma som våra profiler

def sample_params(r: random.Random) -> Dict[str, Any]:
    # Paramrum – justera fritt vid behov. Håller oss nära dina befintliga nycklar.
    return dict(
        use_rsi_filter=r.choice([True, False]),
        rsi_window=r.choice([10, 14, 21]),
        rsi_min=r.choice([25, 30, 35]),
        rsi_max=r.choice([65, 70, 75]),
        breakout_lookback=r.choice([20, 55, 100]),
        exit_lookback=r.choice([10, 20, 50]),
        use_macd_filter=r.choice([False, True]),
        macd_fast=12, macd_slow=26, macd_signal=9,
        use_trend_filter=r.choice([False, True]),
        use_bb_filter=r.choice([False, True]),
        use_stop_loss=r.choice([False, True]),
        stop_mode="pct",
        stop_loss_pct=r.choice([0.05, 0.08, 0.10]),
        atr_window=14,
        atr_mult=r.choice([1.5, 2.0, 2.5]),
        use_atr_trailing=r.choice([False, True]),
        atr_trail_mult=r.choice([1.5, 2.0, 2.5]),
        from_date=DEF_FROM_DATE,
        to_date=None,
    )

def total_return_from_result(res: Dict[str, Any]) -> float | None:
    """Hämta TR robust: först ur summary, annars ur equity-kurvan."""
    # 1) summary.TotalReturn om den finns
    summ = res.get("summary")
    if isinstance(summ, dict):
        tr = summ.get("TotalReturn")
        if tr is not None:
            try:
                return float(tr)
            except Exception:
                pass
    # 2) equity-serien → sista/ första
    try:
        x = res.get("equity") if "equity" in res else res
        eq = extract_equity(x)
        eq = pd.to_numeric(eq, errors="coerce").dropna()
        if len(eq) >= 1:
            return float(eq.iloc[-1] - 1.0)
    except Exception:
        pass
    return None

def run_opt_for_ticker(ticker: str, sims: int, outdir: Path, rng: random.Random):
    outdir.mkdir(parents=True, exist_ok=True)
    log = (outdir / f"{ticker}.log").open("a", encoding="utf-8")
    best: Dict[str, Any] | None = None
    best_tr: float = float("-inf")

    def save_best(bp: Dict[str, Any], btr: float):
        # Spara både ett “resultat” och en profiler-kompatibel fil
        (outdir / f"{ticker}_best.json").write_text(json.dumps(bp, indent=2, ensure_ascii=False), encoding="utf-8")
        prof = {
            "profiles": [{
                "name": f"{ticker} – auto (best of {sims})",
                "ticker": ticker,
                "params": bp["params"],
                "metrics": {"TotalReturn": btr},
            }]
        }
        (outdir / f"{ticker}_profile.json").write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")

    for i in range(1, sims+1):
        params = sample_params(rng)
        try:
            res = W.run_backtest(p={"ticker": ticker, "params": params})
            tr = total_return_from_result(res)
        except Exception as e:
            log.write(f"[{time.strftime('%F %T')}] #{i} ERROR: {type(e).__name__}: {e}\n")
            log.flush()
            continue

        if tr is None:
            log.write(f"[{time.strftime('%F %T')}] #{i} WARN: no TR\n")
            log.flush()
            continue

        if tr > best_tr:
            best_tr = tr
            best = {"ticker": ticker, "params": params, "metrics": {"TotalReturn": best_tr}}
            save_best(best, best_tr)
            log.write(f"[{time.strftime('%F %T')}] #{i} NEW BEST: TR={best_tr:.6f}\n")
            log.flush()

        if i % 500 == 0:
            log.write(f"[{time.strftime('%F %T')}] progress {i}/{sims} (best={best_tr:.6f})\n")
            log.flush()

    log.write(f"[{time.strftime('%F %T')}] DONE {ticker}: sims={sims} best={best_tr:.6f}\n")
    log.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", required=True)
    ap.add_argument("--sims", type=int, default=25000)
    ap.add_argument("--out", default="results/opt_batch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    outdir = Path(args.out)

    tickers = [t.strip() for t in Path(args.tickers_file).read_text(encoding="utf-8").splitlines() if t.strip()]
    for t in tickers:
        run_opt_for_ticker(t, args.sims, outdir, rng)

if __name__ == "__main__":
    main()
