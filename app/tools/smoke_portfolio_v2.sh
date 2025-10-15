#!/usr/bin/env bash
set -euo pipefail
APP="${APP:-/srv/trader/app}"
PY="$APP/.venv/bin/python"
cd "$APP"

"$PY" - <<'PY'
import sys, json
from pathlib import Path
import pandas as pd

ok=True
def fail(msg,e): 
    global ok
    ok=False
    print(f"FAIL {msg}: {type(e).__name__}: {e}")

# A) kärnmoduler
try:
    from app import portfolio_math as PM
    s1=pd.Series([100,110,120], index=pd.date_range("2020-01-01", periods=3))
    s2=pd.Series([100, 90,100], index=s1.index)
    ew=PM.equal_weight_rebalanced([s1,s2])
    print("PASS portfolio_math.equal_weight_rebalanced:", float(ew.iloc[-1]))
except Exception as e: fail("portfolio_math", e)

try:
    from app.trade_extract import to_trades_df
    import json
    # Läs första profil och se att to_trades_df inte kraschar
    pf = next(iter(Path("profiles").glob("*.json")), None)
    if pf:
        d=json.loads(pf.read_text(encoding="utf-8"))
        df=to_trades_df(d.get("trades") or d)
        print("PASS trade_extract.to_trades_df: rows=", len(df))
    else:
        print("SKIP trade_extract (inga profiler)")
except Exception as e: fail("trade_extract", e)

try:
    from app.equity_extract import extract_equity
    s=pd.Series([1,1.1,1.2], index=pd.date_range("2021-01-01", periods=3))
    eq=extract_equity(s)
    print("PASS equity_extract.extract_equity: len=", len(eq))
except Exception as e: fail("equity_extract", e)

# B) lines-sidan – enkel statisk kontroll
try:
    txt=Path("pages/7_Portfolio_V2_Lines.py").read_text(encoding="utf-8")
    # exakt en def to_step
    n_to_step = txt.count("def to_step(")
    assert n_to_step == 1, f"förväntade 1 to_step, hittade {n_to_step}"
    # buyhold_equal_weight måste returnera en Serie (grovt tecken: 'return bh')
    assert "def buyhold_equal_weight" in txt and "return bh" in txt, "buyhold_equal_weight saknar return bh"
    print("PASS 7_Portfolio_V2_Lines.py: struktur OK")
except Exception as e: fail("7_Portfolio_V2_Lines.py", e)

sys.exit(0 if ok else 1)
PY
echo "OK."
