#!/usr/bin/env bash
set -euo pipefail
APP=/srv/trader/app
PY="$APP/.venv/bin/python"
export PYTHONPATH="$APP"

"$PY" - <<'PY'
import json
from pathlib import Path
import pandas as pd

from app import btwrap as W
from app.equity_extract import extract_equity
from app.portfolio_math import equal_weight_rebalanced
from app.data_providers import get_ohlcv

def pick_first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def to_dt(x):
    s = extract_equity(x)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.RangeIndex(len(s))
    return s

# 1) equity för alla profiler
eqs = {}
ticks = []
for f in sorted(Path("profiles").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    p = (d.get("profiles") or [])[0]
    t = p.get("ticker") or (p.get("params") or {}).get("ticker")
    res = W.run_backtest(p={"ticker": t, "params": dict(p.get("params") or {})})
    x = pick_first(res.get("equity", None), res.get("summary", None), res)  # <-- säkert
    s = to_dt(x)
    if len(s) > 1:
        eqs[t] = s
        if t not in ticks: ticks.append(t)

if not eqs:
    raise SystemExit("Inga equity-serier hittades.")

# 2) rebalanced-portfölj
port = equal_weight_rebalanced(eqs.values())

# 3) Buy&Hold (lika vikt, instrumentpriser)
start = str(port.index[0].date())
P = []
for t in ticks:
    df = get_ohlcv(ticker=t, start=start, end=None)[["Close"]].dropna()
    P.append((t, df["Close"]))

idx = None
for _, s in P:
    idx = s.index if idx is None else idx.intersection(s.index)
if idx is None or len(idx)==0:
    raise SystemExit("Kunde inte hitta gemensam period för priser.")

PX = pd.concat([s.reindex(idx) for _, s in P], axis=1)
PX.columns = [t for t,_ in P]
PX = PX.dropna()
bh = (PX / PX.iloc[0]).mean(axis=1)

# 4) Index
idxdf = get_ohlcv("OMXS30GI", start=start, end=None)[["Close"]].dropna()
idxdf = idxdf.reindex(bh.index).dropna()
idx_curve = idxdf["Close"] / idxdf["Close"].iloc[0]

def lvl(s): 
    return float(s.iloc[-1]) if s is not None and len(s) else float("nan")

print(f"COMMON: {bh.index[0].date()} → {bh.index[-1].date()}  rows={len(bh)}")
print(f"BH_final ≈ {lvl(bh):.4f}")
print(f"RB_final ≈ {lvl(port.reindex(bh.index).dropna()):.4f}")
for t in PX.columns:
    r = PX[t].iloc[-1]/PX[t].iloc[0]
    print(f"  {t:8s} BH_ratio(last/first) ≈ {r:.4f}")
print(f"Index(OMXS30GI)_final ≈ {lvl(idx_curve):.4f}")
PY
