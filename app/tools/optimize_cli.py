from __future__ import annotations
import sys, pathlib, json, math, argparse, time, random, statistics as stats
from datetime import date, timedelta, datetime
from typing import Dict, Any, List, Tuple

# Importväg till app-roten
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from app.data_providers import get_ohlcv as GET_OHLCV
from backtest import run_backtest as RUN_BT

def ur(rng: random.Random, a: float, b: float) -> float:
    return a + (b - a) * rng.random()

def make_params(rng: random.Random) -> Dict[str, Any]:
    use_trend_filter   = bool(rng.getrandbits(1))
    use_macd_filter    = bool(rng.getrandbits(1))
    use_bb_filter      = bool(rng.getrandbits(1))
    use_stop_loss      = bool(rng.getrandbits(1))
    use_atr_trailing   = bool(rng.getrandbits(1))
    trend_ma_type      = rng.choice(["SMA","EMA"])

    return {
        "use_rsi_filter": True,
        "rsi_window": rng.randint(8, 32),
        "rsi_min": ur(rng, 5.0, 35.0),
        "rsi_max": ur(rng, 60.0, 85.0),

        "use_trend_filter": use_trend_filter,
        "trend_ma_type": trend_ma_type,
        "trend_ma_window": rng.randint(20, 200),

        "breakout_lookback": rng.randint(20, 120),
        "exit_lookback":     rng.randint(10, 60),

        "use_macd_filter": use_macd_filter,
        "macd_fast":   rng.randint(8, 16),
        "macd_slow":   rng.randint(18, 30),
        "macd_signal": rng.randint(8, 14),

        "use_bb_filter": use_bb_filter,
        "bb_window": rng.randint(15, 30),
        "bb_nstd":   ur(rng, 1.6, 2.4),
        "bb_min":    ur(rng, 0.0, 0.8),

        "use_stop_loss": use_stop_loss,
        "stop_mode": rng.choice(["pct","atr"]),
        "stop_loss_pct": ur(rng, 0.03, 0.20),

        "atr_window": rng.randint(10, 20),
        "atr_mult":   ur(rng, 1.2, 3.2),

        "use_atr_trailing": use_atr_trailing,
        "atr_trail_mult":   ur(rng, 1.2, 3.5),
    }

def score(metrics: Dict[str, Any]) -> float:
    # Högre TR och SharpeD är bra. Mindre (mer positiv) MaxDD är bra (MaxDD är negativt).
    tr = float(metrics.get("TotalReturn") or 0.0)
    sh = float(metrics.get("SharpeD") or 0.0)
    mdd = float(metrics.get("MaxDD") or 0.0)  # negativt tal, mindre negativt är bättre
    # kombinationspoäng (justera vikter vid behov)
    return 2.0*tr + 1.0*sh + 0.5*(-mdd)

def five_year_window_ends_today() -> Tuple[str,str]:
    end = date.today()
    start = end - timedelta(days=365*5 + 2)  # lite slack för helg/kalender
    return (start.isoformat(), end.isoformat())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", type=str, help="Ticker, t.ex. 'GETI B' eller 'VOLV-B.ST'")
    ap.add_argument("--sims", type=int, default=1000, help="Antal simuleringar (default 1000)")
    ap.add_argument("--seed", type=int, default=42, help="Seed för RNG")
    ap.add_argument("--from", dest="from_date", type=str, default=None, help="Startdatum (YYYY-MM-DD). Default: 5 år bakåt från idag")
    ap.add_argument("--to", dest="to_date", type=str, default=None, help="Slutdatum (YYYY-MM-DD). Default: idag")
    ap.add_argument("--outdir", type=str, default="profiles", help="Katalog att spara JSON i")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    # Datumintervall (5 år bakåt om inget angivet)
    if args.from_date and args.to_date:
        start, end = args.from_date, args.to_date
    else:
        start, end = five_year_window_ends_today()

    ticker = args.ticker.strip()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{ticker}.json"

    # 1) Hämta OHLCV EN gång
    df = GET_OHLCV(ticker=ticker, start=start, end=end)
    if df is None or len(getattr(df, "index", [])) == 0:
        raise RuntimeError(f"Tomt OHLCV för {ticker} mellan {start}–{end}")

    best: List[Tuple[float, Dict[str,Any], Dict[str,Any]]] = []  # (score, params, metrics)

    t0 = time.time()
    print(f"[info] Kör {args.sims} simuleringar för {ticker} ({start}–{end}) …")
    last_flush = time.time()

    for i in range(1, args.sims+1):
        params = make_params(rng)
        # injicera datum i params så backtestet vet perioden
        params["from_date"] = start
        params["to_date"]   = end

        try:
            res = RUN_BT(df, params)  # direkt mot backtest.run_backtest(df, p)
            metrics = res.get("summary", {}) if isinstance(res, dict) else {}
            s = score(metrics)
            # håll en liten topplista (topp 16) för effektiv sort
            best.append((s, params, metrics))
            if len(best) > 16:
                best.sort(key=lambda x: x[0], reverse=True)
                best = best[:16]
        except Exception as e:
            # ignorera enstaka fel; visa bara minimal hint ibland
            if (i % 200) == 0:
                print(f"[warn] sim {i} fel: {type(e).__name__}: {e}")

        # progress
        if (i % 100) == 0 or (time.time() - last_flush) > 5:
            dt = time.time() - t0
            rps = i / dt if dt > 0 else 0.0
            print(f"[progress] {i}/{args.sims} sims  ({rps:.1f} runs/s)")
            last_flush = time.time()

    # sortera och välj topp 3
    best.sort(key=lambda x: x[0], reverse=True)
    top = best[:3]

    names = ["conservative", "balanced", "aggressive"]
    profiles = []
    for idx, (s, p, m) in enumerate(top):
        profiles.append({
            "name": f"{ticker} – {names[idx]}",
            "ticker": ticker,
            "params": p,
            "metrics": m,
        })

    payload = {"profiles": profiles}
    outfile.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dt = time.time() - t0
    print(f"[done] Skrev {len(profiles)} profiler → {outfile} (totalt {dt:.1f}s)")

if __name__ == "__main__":
    main()
