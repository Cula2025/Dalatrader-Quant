#!/usr/bin/env bash
set -euo pipefail
cd /srv/trader/app || { echo "[FEL] Hittar inte /srv/trader/app"; exit 1; }

PY="./.venv/bin/python"; [ -x "$PY" ] || PY="python3"

# 1) Hitta backtracker-fil
FILE="pages/1_Backtest.py"
if [ ! -f "$FILE" ]; then
  FILE="$(ls -1d pages/*Backtest*.py pages/*Backtrack*.py 2>/dev/null | head -1 || true)"
fi
[ -n "${FILE:-}" ] && [ -f "$FILE" ] || { echo "[FEL] Hittar ingen Backtracker-fil i pages/"; exit 1; }
echo "[info] Backtracker-fil: $FILE"

# 2) Backup
mkdir -p backups
cp -v "$FILE" "backups/$(basename "$FILE").bak_$(date +%F_%H%M%S)"

# 3) Patch (skicka in BT_FILE till Python-processen!)
BT_FILE="$FILE" "$PY" - <<'PY'
import os, re
from pathlib import Path

p = Path(os.environ["BT_FILE"])
if not p.is_file():
    raise SystemExit(f"[FEL] BT_FILE pekar inte på fil: {p}")

src = p.read_text(encoding="utf-8")
orig = src

# 3.1) Säkerställ 's = st.session_state' direkt efter 'import streamlit as st'
lines = src.splitlines(keepends=True)
for i, ln in enumerate(lines[:200]):
    if re.match(r'^\s*import\s+streamlit\s+as\s+st\b', ln):
        block_after = "".join(lines[i+1:i+6])
        if "s = st.session_state" not in block_after:
            lines.insert(i+1, "s = st.session_state\n")
        break
src = "".join(lines)

# 3.2) Kommentera ut grafer (vi återinför senare när UI är klart)
for fn in ("line_chart","altair_chart","area_chart","bar_chart"):
    src = re.sub(rf'(^\s*)(st\.{fn}\s*\()', r'\1# (disabled) \2', src, flags=re.M)

# 3.3) Byt 2 kolumner till 4 kolumner
src = re.sub(r'st\.columns\(\s*2\s*\)', 'st.columns(4)', src)

# 3.4) Normalisera _RUNBT(ticker, params) -> _RUNBT({"ticker": ticker, "params": params})
src = re.sub(
    r'_RUNBT\s*\(\s*([A-Za-z_][\w\.]*)\s*,\s*([A-Za-z_][\w\.]*)\s*\)',
    r'_RUNBT({"ticker": \1, "params": \2})',
    src
)

if src != orig:
    p.write_text(src, encoding="utf-8")
    print("[ok] Patchad:", p)
else:
    print("[info] Inga ändringar behövdes i", p)
PY

# 4) Syntaxkoll
"$PY" -m py_compile "$FILE" && echo "[ok] Syntax OK"

# 5) Restart UI
sudo systemctl restart trader-ui.service
echo "[done] Ladda om Backtracker-sidan."
