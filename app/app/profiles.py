from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

PROFILES_FP = Path("outputs/profiles/backtest_profiles.json")

def _ensure_dir() -> None:
    PROFILES_FP.parent.mkdir(parents=True, exist_ok=True)

def load_profiles() -> List[Dict[str, Any]]:
    _ensure_dir()
    if not PROFILES_FP.exists() or PROFILES_FP.stat().st_size == 0:
        return []
    try:
        text = PROFILES_FP.read_text(encoding="utf-8", errors="ignore")
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def list_profile_names() -> List[str]:
    return [p.get("name","<unnamed>") for p in load_profiles()]

def upsert_profile(profile: Dict[str, Any]) -> None:
    _ensure_dir()
    data = load_profiles()
    name = str(profile.get("name") or "").strip()
    if not name:
        raise ValueError("Profil saknar 'name'.")

    for i, p in enumerate(data):
        if str(p.get("name","")).strip() == name:
            data[i] = profile
            break
    else:
        data.append(profile)

    PROFILES_FP.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def delete_profile(name: str) -> bool:
    _ensure_dir()
    data = load_profiles()
    new_data = [p for p in data if str(p.get("name","")).strip() != str(name).strip()]
    if len(new_data) == len(data):
        return False
    PROFILES_FP.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return True
