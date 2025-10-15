from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

# motor
try:
    from app.backtracker import run_backtest as RUN
    motor = "backtracker"
except Exception:
    from app.btwrap import run_backtest as RUN
    motor = "btwrap"

from app.trades_util import trades_df_from_result

def best_profile(d: dict):
    ps = d.get("profiles") or []
    if not ps: return None
    return max(ps, key=lambda p: (p.get("metrics") or {}).get("TotalReturn", float("-inf")))

files = sys.argv[1:] or [str(p) for p in sorted(Path("profiles").glob("*.json"))]
print(f"Motor: {motor}")
for f in files:
    print("\n=== FILE:", Path(f).name, "===")
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
    except Exception as e:
        print("  READ ERROR:", e); continue
    bp = best_profile(d)
    if not bp:
        print("  No profiles."); continue
    params = dict((bp.get("params") or {}))
    ticker = bp.get("ticker") or params.get("ticker")
    name   = bp.get("name")
    print("  ticker:", ticker, "| profile:", name)
    res = RUN(p={"ticker": ticker, "params": params})
    trades = res.get("trades")
    print("  trades type:", type(trades).__name__)
    if trades is None:
        print("  trades: None"); continue
    try:
        df = trades_df_from_result(res, ticker=ticker, profile_name=name)
    except Exception as e:
        print("  PARSE ERROR:", e); continue
    if df.empty:
        print("  Parsed DataFrame EMPTY (date not recognized).")
        # show raw head
        try:
            raw = pd.DataFrame(trades)
            print("  Raw head:\n", raw.head(3))
            print("  Raw columns:", list(raw.columns))
        except Exception:
            pass
        continue
    print("  OK:", len(df), "rows")
    print("  Index:", type(df.index).__name__, "| first:", df.index.min(), "| last:", df.index.max())
    print("  Columns:", list(df.columns))
    out_raw = Path("debug")/f"trades_{Path(f).stem}_raw.csv"
    out_par = Path("debug")/f"trades_{Path(f).stem}_parsed.csv"
    try:
        pd.DataFrame(trades).to_csv(out_raw, index=False)
    except Exception:
        pass
    df.to_csv(out_par)
    print("  Saved:", out_par)
