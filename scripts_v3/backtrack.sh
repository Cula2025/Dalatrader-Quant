#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   backtrack.sh "ATCO B" [--profile N] [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--set key=value ...]

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <TICKER> [--profile N] [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--set k=v ...]" >&2
  exit 1
fi
TICKER="$1"; shift || true
PROFILE_IDX="0"
FROM_DATE=""
TO_DATE=""
OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE_IDX="${2:-0}"; shift 2;;
    --from) FROM_DATE="${2:-}"; shift 2;;
    --to)   TO_DATE="${2:-}"; shift 2;;
    --set)  OVERRIDES+=("${2:-}"); shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILES_DIR="$ROOT/profiles_v3"
RUNS_DIR="$ROOT/runs"
TS="$(date +%Y%m%d_%H%M%S)"
SAFE_TICKER="$(printf '%s' "$TICKER" | tr ' /' '__')"
OUT_DIR="$RUNS_DIR/${SAFE_TICKER}_${TS}"
mkdir -p "$OUT_DIR"

PY_BIN="$ROOT/.venv/bin/python"
[[ -x "$PY_BIN" ]] || PY_BIN="$(command -v python3 || command -v python)"

# OVERRIDES → JSON array via jq (om jq finns), annars enkel lista
if command -v jq >/dev/null 2>&1; then
  OVERRIDES_JSON="$(printf '%s\n' "${OVERRIDES[@]:-}" | jq -R . | jq -s .)"
else
  # Fallback: bygg en JSON-array manuellt (utan specials)
  OVERRIDES_JSON="[$(printf '"%s",' "${OVERRIDES[@]:-}" | sed 's/,$//')]"
fi

# Kör Python med *citerad* heredoc och env-variabler
ROOT="$ROOT" OUT_DIR="$OUT_DIR" TICKER="$TICKER" PROFILE_IDX="$PROFILE_IDX" \
FROM_DATE="$FROM_DATE" TO_DATE="$TO_DATE" PROFILES_DIR="$PROFILES_DIR" \
OVERRIDES_JSON="$OVERRIDES_JSON" \
"$PY_BIN" - <<'PY'
import os, sys, json, glob, math
from datetime import date
from typing import Any, Dict, List, Tuple
import importlib
import pandas as pd

ROOT         = os.environ["ROOT"]
PROFILES_DIR = os.environ["PROFILES_DIR"]
OUT_DIR      = os.environ["OUT_DIR"]
TICKER       = os.environ["TICKER"]
PROFILE_IDX  = int(os.environ["PROFILE_IDX"])
FROM_DATE    = os.environ.get("FROM_DATE") or ""
TO_DATE      = os.environ.get("TO_DATE") or ""
OVERRIDES    = json.loads(os.environ.get("OVERRIDES_JSON","[]"))

# Läs profiler för vald ticker, sortera per fil-mtime (desc)
candidates: List[Tuple[float,str,int,Dict[str,Any]]] = []
for path in sorted(glob.glob(os.path.join(PROFILES_DIR, "*.json"))):
    try:
        mtime = os.path.getmtime(path)
        data = json.load(open(path, "r", encoding="utf-8"))
        for idx, prof in enumerate(data.get("profiles", [])):
            if prof.get("ticker") == TICKER:
                candidates.append((mtime, path, idx, prof))
    except Exception:
        pass
candidates.sort(key=lambda x: x[0], reverse=True)

if not candidates:
    print(f"[ERR] No profiles for ticker {TICKER} in {PROFILES_DIR}", file=sys.stderr)
    sys.exit(3)

if PROFILE_IDX < 0 or PROFILE_IDX >= len(candidates):
    print(f"[ERR] profile index {PROFILE_IDX} out of range (0..{len(candidates)-1})", file=sys.stderr)
    sys.exit(4)

_, src_file, src_idx, prof = candidates[PROFILE_IDX]
params = dict(prof.get("params", {}))  # copy

# Datum-override
if FROM_DATE:
    params["from_date"] = FROM_DATE
if TO_DATE:
    params["to_date"]  = TO_DATE

# --set k=v overrides (best-effort typning)
def parse_value(s: str):
    l = s.lower()
    if l in ("true","false"): return l=="true"
    try:
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return s

for kv in OVERRIDES:
    if isinstance(kv, str) and "=" in kv:
        k, v = kv.split("=", 1)
        params[k.strip()] = parse_value(v.strip())

# Datum krävs
from_dt = params.get("from_date")
to_dt   = params.get("to_date")
if not from_dt or not to_dt:
    print("[ERR] params missing from_date/to_date; supply with --from/--to or add to profile", file=sys.stderr)
    sys.exit(5)

def parse_d(d):
    y,m,dd = [int(x) for x in d.split("-")]
    return date(y,m,dd)

start = parse_d(from_dt)
end   = parse_d(to_dt)

# Kör motor
get_ohlcv = importlib.import_module("app_v3.data_provider_v3").get_ohlcv
bt        = importlib.import_module("app_v3.bt_core_v3")

df = get_ohlcv(ticker=TICKER, start=start, end=end)
res = bt.run_backtest(df, params=params)

equity  = res.get("equity")
metrics = res.get("metrics", {})

# Spara artefakter
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

if isinstance(equity, pd.Series):
    pd.DataFrame({
        "Date": equity.index.strftime("%Y-%m-%d"),
        "Strategy": equity.values
    }).to_csv(os.path.join(OUT_DIR, "equity.csv"), index=False)

final_prof = {
    "name": prof.get("name", f"{TICKER} – backtrack"),
    "ticker": TICKER,
    "params": params,
    "metrics": metrics,
}
with open(os.path.join(OUT_DIR, "final_profile.json"), "w", encoding="utf-8") as f:
    json.dump({"profiles": [final_prof]}, f, ensure_ascii=False, indent=2)

# Summering
bt_x  = metrics.get("TotalReturn")
bh_x  = metrics.get("BuyHold")
sharpe= metrics.get("SharpeD")
mdd   = metrics.get("MaxDD")
trd   = metrics.get("Trades")
print(f"[OK] {TICKER}  src={os.path.basename(src_file)}[idx={src_idx}]  BTx={bt_x:.6f}  BHx={bh_x:.6f}  SharpeD={sharpe:.3f}  MaxDD={mdd:.3f}  Trades={int(trd) if trd is not None else 'NA'}")
print(f"[OUT] {OUT_DIR}")
PY
