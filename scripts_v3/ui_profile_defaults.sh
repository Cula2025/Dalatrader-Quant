#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   ui_profile_defaults.sh "ATCO B" [--profile N]
# Prints JSON for exactly ONE profile: params for UI defaults (+ metrics).

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <TICKER> [--profile N]" >&2
  exit 1
fi
TICKER="$1"; shift || true
PROFILE_IDX="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE_IDX="${2:-0}"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILES_DIR="$ROOT/profiles_v3"

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: 'jq' saknas (sudo apt-get install -y jq)" >&2
  exit 2
fi

shopt -s nullglob
files=( "$PROFILES_DIR"/*.json )
shopt -u nullglob
(( ${#files[@]} > 0 )) || { echo '{"error":"no_profiles"}'; exit 0; }

# samla kandidat-profiler för tickern med filens mtime
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
for f in "${files[@]}"; do
  # mtime
  if stat --version >/dev/null 2>&1; then
    epoch=$(stat -c %Y "$f"); human=$(stat -c %y "$f" | cut -d'.' -f1)
  else
    epoch=$(stat -f %m "$f"); human=$(date -r "$epoch" +"%Y-%m-%d %H:%M:%S")
  fi
  jq --arg T "$TICKER" --arg F "$f" --arg E "$epoch" --arg H "$human" '
    .profiles
    | to_entries[]
    | select(.value.ticker == $T)
    | {
        epoch: ($E|tonumber),
        file_mtime: $H,
        file: $F,
        profile_idx: (.key|tonumber),
        ticker: .value.ticker,
        name: .value.name,
        params: .value.params,
        metrics: .value.metrics
      }
  ' "$f" >> "$tmp"
done

# sortera (senast först)
count=$(jq -s 'length' "$tmp")
if [[ "$count" -eq 0 ]]; then
  echo '{"error":"no_profiles_for_ticker"}'
  exit 0
fi

# välj profil (index i den sorterade listan)
sel=$(jq -s "sort_by(-.epoch) | .[${PROFILE_IDX}]" "$tmp")
if [[ "$sel" == "null" ]]; then
  echo '{"error":"profile_index_out_of_range"}'
  exit 0
fi
echo "$sel"
