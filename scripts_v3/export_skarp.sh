#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   export_skarp.sh "ATCO B" [RUN_DIR]
# If RUN_DIR is omitted, the latest runs/<TICKER>_* will be used.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <TICKER> [RUN_DIR]" >&2
  exit 1
fi

TICKER="$1"; shift || true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAFE_TICKER="$(printf '%s' "$TICKER" | tr ' /' '__')"

RUN_DIR="${1-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(ls -dt "$ROOT"/runs/${SAFE_TICKER}_* 2>/dev/null | head -n1 || true)"
fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "[ERR] Ingen run-katalog hittad för '$TICKER' i $ROOT/runs/" >&2
  exit 2
fi

SRC="$RUN_DIR/final_profile.json"
DST_DIR="$ROOT/portfolio_v3/active"
DST="$DST_DIR/${SAFE_TICKER}.final.json"

if [[ ! -f "$SRC" ]]; then
  echo "[ERR] Saknar $SRC" >&2
  exit 3
fi

mkdir -p "$DST_DIR"
cp -f "$SRC" "$DST"
echo "[OK] Exported: $DST"
