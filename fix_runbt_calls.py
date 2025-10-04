from pathlib import Path
import re, sys

ROOT = Path("/srv/trader/app")
targets = [
    ROOT / "pages/0_Optimizer.py",
    ROOT / "pages/1_Backtest.py",
]

pattern = re.compile(
    r"""_RUNBT\s*\(\s*        # _RUNBT(
        ([^)]+)               # capture all args inside parens
    \)""",
    re.VERBOSE | re.DOTALL
)

def rewrite_call(args_src: str) -> str:
    """
    Konverterar vilka varianter som helst av _RUNBT(...) till:
      _RUNBT(p={**params, "ticker": ticker, "from_date": from_date, "to_date": to_date})
    Vi försöker extrahera vettiga variabelnamn som finns i filen. Om både
    from_date/to_date saknas, testar vi också på 'fd'/'td' eller 'start'/'end'.
    """
    # Detta är robustare än "regex by position". Vi genererar en standardmall.
    return '_RUNBT(p={**params, "ticker": ticker, "from_date": from_date if "from_date" in locals() else (fd if "fd" in locals() else (start if "start" in locals() else s.get("from_date"))), "to_date": to_date if "to_date" in locals() else (td if "td" in locals() else (end if "end" in locals() else s.get("to_date")))})'

changed_any = False
for path in targets:
    if not path.exists():
        print(f"[skip] {path} saknas")
        continue

    src = path.read_text(encoding="utf-8")
    # Spåra hur många träffar
    n = 0
    def _repl(m):
        nonlocal n
        n += 1
        args_src = m.group(1)
        return rewrite_call(args_src)

    new_src = pattern.sub(_repl, src)

    if n > 0:
        backup = path.with_suffix(path.suffix + ".bak_runbt_fix")
        backup.write_text(src, encoding="utf-8")
        path.write_text(new_src, encoding="utf-8")
        print(f"[ok] Patchade {path.name}: {n} _RUNBT()-anrop uppdaterade (backup: {backup.name})")
        changed_any = True
    else:
        print(f"[info] {path.name}: Inga _RUNBT()-anrop att patcha")

if not changed_any:
    print("[info] Ingen fil behövde ändras.")
