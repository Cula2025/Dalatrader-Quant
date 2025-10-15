from __future__ import annotations
import sys, json
from pathlib import Path
from typing import Iterable
import pandas as pd

# Motor via wrapper (fungerar för både Optimizer/Backtest)
from app import btwrap as W
from app.data_providers import get_ohlcv
from app.equity_extract import extract_equity

TOL = 1e-6

def roughly_equal(a: float|None, b: float|None, tol=TOL) -> bool:
    if a is None or b is None: return False
    return abs(a-b) <= max(tol, (abs(a)+abs(b))*1e-9)

def iter_inputs(argv: list[str]) -> Iterable[Path]:
    if not argv:
        yield from sorted(Path("profiles").glob("*.json"))
    else:
        for a in argv:
            p = Path(a)
            if p.is_dir(): yield from sorted(p.glob("*.json"))
            elif p.is_file(): yield p

def calc_bh(ticker: str, from_date: str|None, to_date: str|None):
    df = get_ohlcv(ticker=ticker, start=None, end=None)
    if df is None or df.empty:
        return None, "inga data"
    sdf = df
    if from_date: sdf = sdf.loc[str(from_date):]
    if to_date:   sdf = sdf.loc[:str(to_date)]
    if len(sdf) < 2:
        return None, f"slice<2 rader (first={df.index[0].date()} last={df.index[-1].date()})"
    bh = float(sdf["Close"].iloc[-1]) / float(sdf["Close"].iloc[0]) - 1.0
    return bh, None

def main(argv: list[str]) -> int:
    files = list(iter_inputs(argv))
    total = passed = failed = 0
    for path in files:
        d = json.loads(path.read_text(encoding="utf-8"))
        profs = d.get("profiles") or []
        print(f"\n== {path} :: {len(profs)} profiler ==")
        for i,p in enumerate(profs,1):
            total += 1
            name   = p.get("name")
            ticker = p.get("ticker") or (p.get("params") or {}).get("ticker")
            params = dict(p.get("params") or {})
            fd, td = params.get("from_date"), params.get("to_date")

            # BH med robust skivning
            bh_calc, err = calc_bh(ticker, fd, td)

            # TR via motor + equity-extrakt
            res = W.run_backtest(p={"ticker": ticker, "params": params})
            x   = res.get("equity") if res.get("equity") is not None else (res.get("summary") if res.get("summary") is not None else res)
            eq  = extract_equity(x)
            tr_calc = (float(eq.iloc[-1]) - 1.0) if len(eq) else None

            met = p.get("metrics") or {}
            tr_facit = met.get("TotalReturn")
            bh_facit = met.get("BuyHold")

            if bh_calc is None:
                print(f"  [{i}/3] {name:<20}  SKIP ({err})")
                continue

            ok_tr = (tr_facit is None) or roughly_equal(tr_calc, tr_facit)
            ok_bh = (bh_facit is None) or roughly_equal(bh_calc, bh_facit)
            status = "PASS" if (ok_tr and ok_bh) else "FAIL"
            if status == "PASS": passed += 1
            else: failed += 1

            print(f"  [{i}/3] {name:<20}  TR(calc)={None if tr_calc is None else round(tr_calc,6)}  "
                  f"facit={tr_facit}  |  BH(calc)={bh_calc:.6f}  facit={bh_facit}  -> {status}")

    print(f"\n=== SAMMANFATTNING ===\nProfiler totalt: {total}\nPASS: {passed}\nFAIL: {failed}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
