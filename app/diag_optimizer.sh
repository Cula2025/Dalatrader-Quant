#!/usr/bin/env bash
set -euo pipefail
cd /srv/trader/app
# Use venv if present
[ -f .venv/bin/activate ] && . .venv/bin/activate

python - <<'PY'
import json, os, glob, traceback, random
from pathlib import Path

print("== DIAG: locate ticker & known-good params from profiles/ ==")
pdir = Path("profiles")
latest = None
if pdir.exists():
    files = sorted(pdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest = files[0] if files else None
print(" latest:", latest)

ticker = os.environ.get("TICKER","GETI B")
params_from_file = None
if latest:
    try:
        data = json.loads(Path(latest).read_text(encoding="utf-8"))
        profs = data.get("profiles",[])
        if profs:
            ticker = (profs[0].get("ticker") or ticker).strip()
            params_from_file = (profs[0].get("params") or {}).copy()
            print(f" using ticker from file: {ticker}")
    except Exception as e:
        print("[warn] could not read profiles:", type(e).__name__, e)

# 1) Smoke test: run engine with known-good params (from file)
print("\n== SMOKE: run_backtest with known-good params ==")
try:
    from app.btwrap import run_backtest as RUNBT
    P = {"ticker": ticker, "params": params_from_file or {
        "from_date":"2020-10-01","to_date":"2025-10-01",
        "use_rsi_filter": True, "rsi_window": 14, "rsi_min": 30.0, "rsi_max": 70.0,
        "use_trend_filter": False, "trend_ma_type": "SMA", "trend_ma_window": 200,
        "breakout_lookback": 55, "exit_lookback": 20,
        "use_macd_filter": False, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "use_bb_filter": False, "bb_window": 20, "bb_nstd": 2.0, "bb_min": 0.0,
        "use_stop_loss": False, "stop_mode":"pct", "stop_loss_pct": 0.10,
        "atr_window": 14, "atr_mult": 2.0, "use_atr_trailing": False, "atr_trail_mult": 2.0
    }}
    res = RUNBT(p=P)
    summ = res.get("summary",{})
    print("[OK] Engine returned summary:", {k:summ.get(k) for k in ("TotalReturn","SharpeD","MaxDD","Trades","Bars")})
except Exception as e:
    print("[ERR] Engine failed on smoke test:", type(e).__name__, e)
    print(traceback.format_exc())

# 2) Random sampler: show first 10 outcomes + any errors
def draw(rng, safe=False):
    ur = lambda a,b: float(a + (b-a)*rng.random())
    d = {
        "use_rsi_filter": True,
        "rsi_window": rng.randint(5,35),
        "rsi_min": ur(5.0,35.0),
        "rsi_max": ur(60.0,85.0),

        "use_trend_filter": bool(rng.choice([True,False])),
        "trend_ma_type": rng.choice(["SMA","EMA"]),
        "trend_ma_window": rng.randint(20,200),

        "breakout_lookback": rng.randint(20,120),
        "exit_lookback":     rng.randint(10,60),

        "use_macd_filter": bool(rng.choice([True,False])),
        "macd_fast":   rng.randint(8,16),
        "macd_slow":   rng.randint(18,30),
        "macd_signal": rng.randint(8,14),

        "use_bb_filter": bool(rng.choice([True,False])),
        "bb_window": rng.randint(15,30),
        "bb_nstd":   ur(1.6,2.4),
        "bb_min":    ur(0.0,0.8),

        "use_stop_loss": bool(rng.choice([True,False])),
        "stop_mode": rng.choice(["pct","atr"]),
        "stop_loss_pct": ur(0.03,0.20),

        "atr_window": rng.randint(10,20),
        "atr_mult":   ur(1.2,3.2),

        "use_atr_trailing": bool(rng.choice([True,False])),
        "atr_trail_mult":   ur(1.2,3.5),
    }
    if safe:
        # Safer subset to avoid exotic combos temporarily
        d["use_macd_filter"] = False
        d["use_bb_filter"] = False
        d["use_stop_loss"] = False
        d["use_atr_trailing"] = False
    return d

print("\n== RANDOM SAMPLER (10 draws, SAFE MODE OFF then ON) ==")
rng = random.Random(42)
from app.btwrap import run_backtest as RUNBT

def try_10(safe):
    ok, errs = 0, []
    for i in range(10):
        p = draw(rng, safe=safe)
        # inherit dates from file if present
        if params_from_file:
            fd = params_from_file.get("from_date"); td = params_from_file.get("to_date")
            if fd: p["from_date"] = fd
            if td: p["to_date"]   = td
        else:
            p.setdefault("from_date","2020-10-01")
            p.setdefault("to_date","2025-10-01")
        try:
            res = RUNBT(p={"ticker": ticker, "params": p})
            summ = res.get("summary",{})
            ok += 1
            print(f"  [OK {ok}] TR={summ.get('TotalReturn'):.3f} DD={summ.get('MaxDD')} Sh={summ.get('SharpeD')}")
        except Exception as e:
            errs.append((type(e).__name__, str(e)))
    return ok, errs

print("-- SAFE=FALSE --")
ok1, err1 = try_10(False)
print(f"   ok={ok1}, err={len(err1)}")
if err1:
    print("   first errors:")
    for t,m in err1[:3]:
        print("    •", t, ":", m)

print("-- SAFE=TRUE --")
ok2, err2 = try_10(True)
print(f"   ok={ok2}, err={len(err2)}")
if err2:
    print("   first errors:")
    for t,m in err2[:3]:
        print("    •", t, ":", m)

print("\n== DONE ==")
PY
