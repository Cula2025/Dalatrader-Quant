#!/usr/bin/env bash
set -euo pipefail

APP=/srv/trader/app
cd "$APP"

# Miljö & tolk
export PYTHONPATH="$APP"
PY="$APP/.venv/bin/python"

# In-parametrar / defaults
TICKERS_FILE="${TICKERS_FILE:-config/tickers_nightly.txt}"
SIMS="${SIMS:-25000}"
SEED="${SEED:-123}"

OUTDIR="results/opt_nightly_$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR" runtime

echo "[start] $(date)  tickers_file=$TICKERS_FILE  sims=$SIMS  seed=$SEED  out=$OUTDIR"

# Torris: visa att vi kan importera app
$PY - <<'PY'
import sys
print("[python] ok", sys.version)
import app  # ska funka tack vare PYTHONPATH
print("[import] app ok")
PY

# Kör batch-optimizern
$PY tools/optimize_batch.py --tickers-file "$TICKERS_FILE" --sims "$SIMS" --out "$OUTDIR" --seed "$SEED" | tee "$OUTDIR/run.log"

echo "[done] $(date)  out -> $OUTDIR"
