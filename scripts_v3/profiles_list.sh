#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   profiles_list.sh [TICKER_FILTER]
#
# Lists profiles found in profiles_v3/*.json, sorted by file mtime (desc).
# Prints: [row#]  YYYY-mm-dd HH:MM  <TICKER>  <FILE>  profile_idx=<IDX>  name="<NAME>"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILES_DIR="$DIR/profiles_v3"
FILTER="${1-}"

# Require jq
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: 'jq' saknas. Installera t.ex.: sudo apt-get install -y jq" >&2
  exit 2
fi

shopt -s nullglob
files=( "$PROFILES_DIR"/*.json )
shopt -u nullglob

if (( ${#files[@]} == 0 )); then
  echo "Inga profiler hittades i $PROFILES_DIR" >&2
  exit 0
fi

# Build lines: EPOCH \t HUMAN \t TICKER \t FILE \t IDX \t NAME
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

for f in "${files[@]}"; do
  # epoch + human mtime for sorting/printing
  if stat --version >/dev/null 2>&1; then
    epoch=$(stat -c %Y "$f")
    human=$(stat -c %y "$f" | cut -d'.' -f1)
  else
    # BSD/mac fallback (not expected here, but kept for safety)
    epoch=$(stat -f %m "$f")
    human=$(date -r "$epoch" +"%Y-%m-%d %H:%M:%S")
  fi

  # enumerate profiles in the file
  jq -r '.profiles | to_entries[] | [.key, .value.ticker, .value.name] | @tsv' "$f" | \
  while IFS=$'\t' read -r idx ticker name; do
    # optional filter on ticker (substring match)
    if [[ -n "$FILTER" ]]; then
      case "$ticker" in
        *"$FILTER"*) ;;  # keep
        *) continue ;;
      esac
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$epoch" "$human" "$ticker" "$f" "$idx" "$name" >> "$tmp"
  done
done

# Sort by epoch desc, then pretty print with a row counter
sort -r -n -k1,1 "$tmp" | awk -F'\t' '
BEGIN { OFS="  " }
{
  row += 1
  epoch=$1; human=$2; ticker=$3; file=$4; idx=$5; name=$6
  printf "[%d] %s  %-8s  %s  profile_idx=%s  name=\"%s\"\n", row, human, ticker, file, idx, name
}
'
