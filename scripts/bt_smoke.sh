#!/usr/bin/env bash
set -euo pipefail
echo "=== Backtester smoke ==="
# venv
if [ -d ".venv" ]; then . .venv/bin/activate; fi
command -v python >/dev/null || { echo "[FAIL] python saknas"; exit 1; }

# 0) Miljö
echo "[info] PY:" $(python -V)
echo "[info] cwd:" $(pwd)

# 1) Moduler & motorfil
python - <<'PY'
import os, importlib, importlib.util, sys
ok=True
try:
    import app.btwrap as W
    print("[OK] import app.btwrap")
    # Försök lista vilken motorfil wrappen laddar (om det står i koden)
    p = os.path.join(os.path.dirname(W.__file__))
    print("[info] btwrap path:", p)
except Exception as e:
    ok=False; print("[FAIL] import btwrap:", e)

try:
    from app.data_providers import get_ohlcv
    print("[OK] import data_providers.get_ohlcv")
except Exception as e:
    ok=False; print("[FAIL] import data_providers:", e)

sys.exit(0 if ok else 2)
PY

# 2) Hitta profiler (GETI B.json)
PROF=""
for d in "/srv/trader/profiles" "/srv/trader/app/profiles"; do
  [ -f "$d/GETI B.json" ] && PROF="$d/GETI B.json" && break
done
if [ -z "$PROF" ]; then
  echo "[FAIL] Hittar inte profilfilen 'GETI B.json' under /srv/trader{,/app}/profiles"
  exit 3
fi
echo "[OK] Profil hittad: $PROF"

# 3) Dataleverans (Börsdata via din provider)
python - <<'PY'
import pandas as pd
from app.data_providers import get_ohlcv
ticker="GETI B"; start="2020-10-05"
df = get_ohlcv(ticker, start=start)
print("[info] OHLCV rows:", len(df), "first:", df.index.min(), "last:", df.index.max())
assert len(df)>200, "för få datapunkter från provider"
print("[PASS] Data-provider OK")
PY

# 4) Kör backtest med profilens Params (conservative index=0 som default)
python - <<'PY'
import json, pandas as pd
from app.btwrap import run_backtest
from app.data_providers import get_ohlcv

ticker="GETI B"; start="2020-10-05"
# Läs profil och plocka första profilen (index 0)
import sys, os
prof_path = os.environ.get("PROF_PATH")
with open(prof_path, "r") as f:
    J = json.load(f)
# stöd både .profiles array och direkt "Params" toppnivå
if isinstance(J, dict) and "profiles" in J and isinstance(J["profiles"], list) and J["profiles"]:
    params = J["profiles"][0].get("Params", {})
else:
    params = J.get("Params", {})

# injicera start om saknas
params = {"start": start, **params}

res = run_backtest({"ticker": ticker, "params": params})
eq = pd.to_numeric(pd.Series(res.get("equity")), errors="coerce").dropna()
tr = res.get("trades", [])
print("[info] eq_points:", len(eq), "trades_len:", (len(tr) if isinstance(tr, list) else "n/a"))
assert len(eq)>200, "equity för kort"
# jämför mot BH grovt
px = get_ohlcv(ticker, start=start)["Close"].dropna()
px = px[px.index>=eq.index.min()]
bh = px/float(px.iloc[0])
bt_x = float(eq.iloc[-1]/eq.iloc[0])
bh_x = float(bh.iloc[-1])
print(f"[info] BT×={bt_x:.4f}  BH×={bh_x:.4f}")
# sanity-gränser (löst hållna)
assert 0.2 < bt_x < 5.0, "BT× orimligt"
assert 0.2 < bh_x < 5.0, "BH× orimligt"
print("[PASS] Backtest körde OK")
PY
