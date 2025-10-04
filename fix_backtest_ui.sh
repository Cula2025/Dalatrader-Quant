#!/usr/bin/env bash
set -euo pipefail

FILE="/srv/trader/app/pages/1_Backtest.py"

if [[ ! -f "$FILE" ]]; then
  echo "[ERR] Hittar inte $FILE" >&2
  exit 1
fi

# 1) Backup
ts=$(date +%F_%H%M%S)
cp -v "$FILE" "${FILE}.bak_ui_${ts}"

# Jobba på temp
tmp="$(mktemp)"
cp "$FILE" "$tmp"

# 2) Säkra s/_st efter "import streamlit as st"
if ! grep -qE '^\s*s\s*=\s*st\.session_state\b' "$tmp"; then
  awk '
    BEGIN{done=0}
    {
      print $0
      if (!done && $0 ~ /^\s*import\s+streamlit\s+as\s+st\b/) {
        print "s = st.session_state"
        print "_st = st"
        done=1
      }
    }
  ' "$tmp" > "${tmp}.ins" && mv "${tmp}.ins" "$tmp"
  echo "[ok] Infogade s/_st."
else
  if ! grep -qE '^\s*_st\s*=\s*st\b' "$tmp"; then
    awk '
      BEGIN{done=0}
      {
        print $0
        if (!done && $0 ~ /^\s*s\s*=\s*st\.session_state\b/) {
          print "_st = st"
          done=1
        }
      }
    ' "$tmp" > "${tmp}.ins" && mv "${tmp}.ins" "$tmp"
    echo "[ok] Infogade _st = st."
  else
    echo "[info] s/_st fanns redan."
  fi
fi

# 3) Dölj översta Resultat/Equity-blocket (mellan header "Resultat" och nästa stora sektion)
awk '
  BEGIN{skip=0}
  /^\s*(st|_st)\.header\(\s*["'\'' ]Resultat["'\'' ]\s*\)/ { skip=1; print "# --- DISABLED: Resultat-block början (borttaget) ---"; next }
  skip==1 && /^\s*(st|_st)\.(sub)?header\(\s*["'\'' ](Parametrar|Trades|Handel|Inställningar|Params)["'\'' ]\s*\)/ {
    skip=0; print "# --- DISABLED: Resultat-block slut ---"; print $0; next
  }
  skip==1 { next }
  { print $0 }
' "$tmp" > "${tmp}.norslt" && mv "${tmp}.norslt" "$tmp"
echo "[ok] Försökte dölja Resultat/Equity-blocket."

# 4) Byt potentiellt dubbla keys -> unika i huvudytan
sed -E -i 's/(key\s*=\s*")[Ff]rom_date(")/\1bt_from_date\2/g' "$tmp"
sed -E -i 's/(key\s*=\s*")[Tt]o_date(")/\1bt_to_date\2/g' "$tmp"
echo "[ok] Uppdaterade from_date/to_date-keys till bt_from_date/bt_to_date."

# 5) Syntaxkontroll
python3 -m py_compile "$tmp"

# 6) Skriv tillbaka
mv -v "$tmp" "$FILE"
echo "[done] Klar. Ladda om Backtest-sidan. Backup: ${FILE}.bak_ui_${ts}"
