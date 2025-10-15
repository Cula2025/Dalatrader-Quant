from __future__ import annotations
import json
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
        # fallback: försök hitta någon profilfil som matchar
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
            # se till att rätt ticker ligger i profilen
            p["ticker"] = t
            profiles.append(p)
        else:
            print(f"⚠️ Hittar ingen best-profil för: {t}")

    payload = {
        "name": name,
        "source": "csv",                   # vi kör på lokala CSV:er i din demo
        "universe": tickers,               # samma universum som profilerna
        "start": "2020-01-01",             # justera om du vill
        "end": "",                         # tom = till senaste
        "rules": {
            "capital": 100000.0,
            "per_trade_pct": 10.0,        # max % per affär
            "max_exposure_pct": 100.0,    # total exponering
            "max_positions": 10           # max antal öppna
        },
        "profiles_mode": "Matcha profiler PER ticker",
        "profiles": profiles               # viktiga biten: en profil per ticker
    }
    return payload

def main():
    # Välj vilka tickers du vill in i portföljen
    tickers = ["SHB A", "ERIC B", "INVE B", "VOLV B"]  # lägg till/ta bort som du vill
    name = "OMX_Demo"

    pf = build_portfolio(name, tickers)
    out = PORTFOLIOS_DIR / f"{name}.json"
    out.write_text(json.dumps(pf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Skrev portfölj: {out}")

if __name__ == "__main__":
    main()

