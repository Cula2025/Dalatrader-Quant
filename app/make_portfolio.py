from __future__ import annotations
import json
from datetime import date
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent
PROFILES_DIR = ROOT / "outputs" / "profiles"
PORTFOLIOS_DIR = ROOT / "storage" / "portfolios"
PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)

def norm(t: str) -> str:
    return t.replace(" ", "_").replace("/", "-")

def load_best(ticker: str) -> Dict[str, Any] | None:
    fp = PROFILES_DIR / f"{norm(ticker)}_best.json"
    if not fp.exists():
        cands = list(PROFILES_DIR.glob(f"*{norm(ticker)}*best*.json"))
        if not cands:
            return None
        fp = cands[0]
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        profs = data.get("profiles", [])
        return profs[0] if profs else None
    except Exception:
        return None

def build_portfolio(name: str, tickers: List[str]) -> Dict[str, Any]:
    profiles: List[Dict[str, Any]] = []
    for t in tickers:
        p = load_best(t)
        if p:
            p["ticker"] = t
            profiles.append(p)
        else:
            print(f"⚠️ Hittar ingen best-profil för: {t}")

    payload = {
        "name": name,
        "source": "csv",
        "universe": tickers,
        "start": "2020-01-01",
        "end": date.today().isoformat(),   # <-- inte tomt längre
        "rules": {
            "capital": 100000.0,
            "per_trade_pct": 10.0,
            "max_exposure_pct": 100.0,
            "max_positions": 10
        },
        "profiles_mode": "Matcha profiler PER ticker",
        "profiles": profiles
    }
    return payload

def main():
    tickers = ["SHB A", "ERIC B", "INVE B", "VOLV B"]
    name = "OMX_Demo"
    pf = build_portfolio(name, tickers)
    out = PORTFOLIOS_DIR / f"{name}.json"
    out.write_text(json.dumps(pf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Skrev portfölj: {out}")

if __name__ == "__main__":
    main()

