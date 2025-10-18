#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   profile_pick.sh "ATCO B" [LIMIT]
# Prints JSON array with up to LIMIT profiles for the ticker,
# sorted by profile file mtime (desc). Default LIMIT=3.
#
# Output fields per item:
#   {
#     "ticker": "...",
#     "name": "...",
#     "file": "profiles_v3/ATCO_B.json",
#     "profile_idx": 0,
#     "file_mtime": "2025-10-17 22:24:00",
#     "metrics": { "TotalReturn": ..., "SharpeD": ..., "MaxDD": ..., "BuyHold": ..., "CAGR": ..., "Trades": ... },
#     "params":  { ... }   # full params from profile
#   }

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <TICKER> [LIMIT]" >&2
  exit 1
fi

TICKER="$1"
LIMIT="${2:-3}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILES_DIR="$ROOT/profiles_v3"

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: 'jq' saknas (sudo apt-get install -y jq)" >&2
  exit 2
fi

shopt -s nullglob
files=( "$PROFILES_DIR"/*.json )
shopt -u nullglob

if (( ${#files[@]} == 0 )); then
  echo "[]" ; exit 0
fi

# Bygg en lista {epoch, human, file, idx, obj}
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT

for f in "${files[@]}"; do
  # mtime
  if stat --version >/dev/null 2>&1; then
    epoch=$(stat -c %Y "$f")
    human=$(stat -c %y "$f" | cut -d'.' -f1)
  else
    epoch=$(stat -f %m "$f")
    human=$(date -r "$epoch" +"%Y-%m-%d %H:%M:%S")
  fi

  # plocka profiler för rätt ticker
  jq --arg T "$TICKER" --arg F "$f" --arg E "$epoch" --arg H "$human" '
    .profiles
    | to_entries[]
    | select(.value.ticker == $T)
    | {
        epoch: ($E|tonumber),
        file_mtime: $H,
        file: $F,
        profile_idx: .key|tonumber,
        ticker: .value.ticker,
        name: .value.name,
        params: .value.params,
        metrics: .value.metrics
      }
  ' "$f" >> "$tmp"
done

# sortera och begränsa, skriv JSON-array
jq -s --argjson limit "$LIMIT" '
  sort_by(-.epoch) | .[:$limit]
' "$tmp"
