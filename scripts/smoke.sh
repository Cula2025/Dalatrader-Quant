#!/usr/bin/env bash
set -euo pipefail

# Hitta Python
PY="${PY:-$(command -v python || true)}"
PY="${PY:-$(command -v python3 || true)}"
[ -z "${PY}" ] && { echo "[fail] Hittar ingen python/python3"; exit 1; }

# Aktivera venv om den finns
if [ -d ".venv" ]; then
  . .venv/bin/activate
  PY="$(command -v python)"
fi

export PYTHONUNBUFFERED=1
export BORS_API_KEY="${BORS_API_KEY:-dummy}"

echo "[1/2] Import-check…"
"$PY" -c "import importlib; [importlib.import_module(m) for m in ['app.portfolio_backtest','app.btwrap']]; print('[ok] Importer funkar')"

echo "[2/2] Mini-BH-test…"
"$PY" -c "import pandas as pd; from app.portfolio_backtest import buyhold_equity_from_close as f; close=pd.Series([100.0,110.0,90.0,120.0], name='Close'); eq=f(close, fee_bps=0.0, slippage_bps=0.0); assert len(eq)==len(close) and abs(eq.iloc[-1]-(close.iloc[-1]/close.iloc[0]))<1e-3; print('[ok] BH ≈ Close-normaliserad')"

echo "[done] Smoke passerad."
