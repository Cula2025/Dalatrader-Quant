set -euo pipefail

FILE="/srv/trader/app/pages/1_Backtest.py"
BACKUP="${FILE}.bak_$(date +%F_%H%M%S)"

echo "[i] Backar upp till: $BACKUP"
cp -v "$FILE" "$BACKUP"

# Kommentera bort vanliga debug-utskrifter som visar rådata i UI
# - st.json(...)
# - st.write(result|equity|trades)
# - st.dataframe(...equity|trades...)
# - (samt table/line_chart/altair_chart för equity/trades om de skulle finnas)
perl -0777 -pe '
  # st.json(result) eller st.json({... "TotalReturn": ...})
  s/^(\s*)(st\.json\s*\(\s*result\s*\).*\n)/$1# UI_SUPPRESS: $2/gm;
  s/^(\s*)(st\.json\s*\(\s*\{\s*["'\'']TotalReturn["'\''].*?\}\s*\).*\n)/$1# UI_SUPPRESS: $2/gms;

  # st.write(result|equity|trades)
  s/^(\s*)(st\.write\s*\(\s*result\s*\).*\n)/$1# UI_SUPPRESS: $2/gm;
  s/^(\s*)(st\.write\s*\(\s*equity\s*\).*\n)/$1# UI_SUPPRESS: $2/gm;
  s/^(\s*)(st\.write\s*\(\s*trades?\s*\).*\n)/$1# UI_SUPPRESS: $2/gm;

  # st.dataframe(equity|trades)
  s/^(\s*)(st\.(?:dataframe|table|line_chart|altair_chart)\s*\(\s*equity[^\n]*\n)/$1# UI_SUPPRESS: $2/gm;
  s/^(\s*)(st\.(?:dataframe|table|line_chart|altair_chart)\s*\(\s*trades?[^\n]*\n)/$1# UI_SUPPRESS: $2/gm;

  # st.json/st.write med equity/trades om de skulle förekomma
  s/^(\s*)(st\.(?:write|json)\s*\(\s*equity[^\n]*\n)/$1# UI_SUPPRESS: $2/gm;
  s/^(\s*)(st\.(?:write|json)\s*\(\s*trades?[^\n]*\n)/$1# UI_SUPPRESS: $2/gm;
' "$FILE" > "${FILE}.tmp"

mv -v "${FILE}.tmp" "$FILE"

# Syntaxcheck i din venv (justera path om din venv ligger annorlunda)
cd /srv/trader/app
./.venv/bin/python -m py_compile pages/1_Backtest.py

# Visa vad som kommenterades (om något)
echo
echo "[i] Följande rader stängdes av:"
grep -n "UI_SUPPRESS" -n pages/1_Backtest.py || echo "(Inga träffar – inget att stänga av)"

echo
echo "[ok] Klart. Ladda om Backtest-sidan i webbläsaren (Shift+Reload om den cachear)."
