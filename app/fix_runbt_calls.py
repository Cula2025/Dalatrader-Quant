from pathlib import Path
import re

ROOT = Path("/srv/trader/app")
targets = [ROOT / "pages/0_Optimizer.py", ROOT / "pages/1_Backtest.py"]

# Matcha _RUNBT( ... ) inkl radbrytningar
pattern = re.compile(r"_RUNBT\s*\((.*?)\)", re.DOTALL)

def repl(m: re.Match) -> str:
    args_src = m.group(1)
    # Lämna orört om det redan är _RUNBT(p= ...)
    if re.search(r"\bp\s*=", args_src):
        return m.group(0)

    # Ersätt alla varianter med en standard: _RUNBT(p={...})
    # Vi försöker använda lokala namn om de finns (ticker/from_date/to_date/params),
    # annars fallbackar vi till session state 's'.
    new = (
        "_RUNBT(p={**(params if 'params' in locals() else (p if 'p' in locals() else {})), "
        "'ticker': (ticker if 'ticker' in locals() else (s.get('ticker') if 's' in locals() else st.session_state.get('ticker'))), "
        "'from_date': (from_date if 'from_date' in locals() else (fd if 'fd' in locals() else (start if 'start' in locals() else (s.get('from_date') if 's' in locals() else st.session_state.get('from_date'))))), "
        "'to_date': (to_date if 'to_date' in locals() else (td if 'td' in locals() else (end if 'end' in locals() else (s.get('to_date') if 's' in locals() else st.session_state.get('to_date')))))"
        "})"
    )
    return new

for path in targets:
    if not path.exists():
        print(f"[skip] {path} saknas")
        continue

    src = path.read_text(encoding="utf-8")
    new_src, count = pattern.subn(repl, src)

    if count > 0:
        backup = path.with_suffix(path.suffix + ".bak_runbt_fix")
        backup.write_text(src, encoding="utf-8")
        path.write_text(new_src, encoding="utf-8")
        print(f"[ok] Patchade {path.name}: {count} st _RUNBT()-anrop uppdaterade (backup: {backup.name})")
    else:
        print(f"[info] {path.name}: Inga _RUNBT()-anrop att patcha")
