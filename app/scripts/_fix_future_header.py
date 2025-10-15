from pathlib import Path
import re

files = [Path(r"app\pages\1_backtest.py"), Path(r"app\run_ui.py")]

for fp in files:
    if not fp.exists():
        print(f"[SKIP] {fp} saknas")
        continue
    txt = fp.read_text(encoding="utf-8", errors="ignore")
    txt = txt.lstrip("\ufeff")  # ta bort ev. BOM

    # plocka bort alla existerande future-rader och env_bootstrap-rader
    future_pat = re.compile(r"^\s*from __future__ import annotations\s*$", re.M)
    env_pat    = re.compile(r"^\s*import app\.env_bootstrap as _env\s*$", re.M)

    has_future = bool(future_pat.search(txt))
    txt = future_pat.sub("", txt)
    txt = env_pat.sub("", txt)

    header = []
    if has_future:
        header.append("from __future__ import annotations")
    header.append("import app.env_bootstrap as _env")

    new_txt = "\n".join(header) + "\n" + txt.lstrip()
    fp.write_text(new_txt, encoding="utf-8")
    print(f"[OK] Fixade header i {fp}")
