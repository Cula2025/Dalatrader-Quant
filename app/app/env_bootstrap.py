from __future__ import annotations
import os
from pathlib import Path

def _parse_env_file(p: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not p.exists():
        return env
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")  # ta bort ev. citattecken
        env[k] = v
    return env

def load_env() -> None:
    """
    Ladda .env fr?n projektroten (C:\\trader) och L?T .env OVERSKRIVA processmilj?n.
    Mappa BORSDATA_KEY <-> BORS_API_KEY och trimma whitespace.
    """
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"

    file_env = _parse_env_file(env_path)

    # 1) Skriv in .env till processmilj?n (OVERRIDE)
    for k, v in file_env.items():
        os.environ[k] = v.strip()

    # 2) Samla ihop nyckel (vilken som helst som finns) och spegla till b?da
    key = (os.environ.get("BORS_API_KEY", "") or os.environ.get("BORSDATA_KEY", "")).strip()
    if key:
        os.environ["BORS_API_KEY"] = key
        os.environ["BORSDATA_KEY"] = key


